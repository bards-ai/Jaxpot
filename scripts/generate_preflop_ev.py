"""Regenerate the preflop_ev.npz lookup table.

For every (hero_hole, vill_hole) pair where the four cards are distinct, this
script enumerates all C(48, 5) = 1,712,304 possible 5-card boards and records
the win and tie fractions for hero. The result is saved as a sorted lookup
table consumed by ``jaxpot.env.poker.evaluator._preflop_lookup_jit``.

Usage::

    python scripts/generate_preflop_ev.py [--out PATH] [--batch B] [--limit N]

Notes
-----
- Runs on whatever JAX backend is available (GPU recommended).
- Memory usage scales with ``--batch``. Default 4 fits in ~1 GB.
- The result is sorted by ``hero_code * 2704 + vill_code`` so that
  ``jnp.searchsorted`` can be used at runtime.
"""

from __future__ import annotations

import argparse
import os
import time
from itertools import combinations

import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np


def _load_evaluator():
    """Load evaluator.py directly to avoid pkg-init circular imports."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "..", "src", "jaxpot", "env", "poker", "evaluator.py")
    spec = importlib.util.spec_from_file_location("_evaluator_direct", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_evaluator_direct"] = mod
    spec.loader.exec_module(mod)
    return mod


_eval_mod = _load_evaluator()
compare_two_players = _eval_mod.compare_two_players
remaining_after_known_any = _eval_mod.remaining_after_known_any

# All C(48, 5) board indices, used to gather from the 48 remaining cards.
COMBS_48_5_NP = np.asarray(list(combinations(range(48), 5)), dtype=np.int32)
NUM_BOARDS = COMBS_48_5_NP.shape[0]
assert NUM_BOARDS == 1_712_304, f"Expected 1712304 boards, got {NUM_BOARDS}"
COMBS_48_5 = jnp.asarray(COMBS_48_5_NP)


def _equity_single(hero_hole: jnp.ndarray, vill_hole: jnp.ndarray):
    """Compute (win_frac, tie_frac) for a single (hero, vill) pair."""
    known = jnp.concatenate([hero_hole, vill_hole])  # (4,)
    rem = remaining_after_known_any(known)  # (48,)
    boards = rem[COMBS_48_5]  # (1712304, 5)
    base_h = jnp.broadcast_to(hero_hole, (NUM_BOARDS, 2))
    base_v = jnp.broadcast_to(vill_hole, (NUM_BOARDS, 2))
    hero7 = jnp.concatenate([base_h, boards], axis=1)
    vill7 = jnp.concatenate([base_v, boards], axis=1)
    res = jax.vmap(compare_two_players, in_axes=(0, 0))(hero7, vill7)
    wins = jnp.sum(res == jnp.int8(-1))
    ties = jnp.sum(res == jnp.int8(0))
    inv_total = jnp.float32(1.0 / NUM_BOARDS)
    return wins.astype(jnp.float32) * inv_total, ties.astype(jnp.float32) * inv_total


_equity_batch_jit = jax.jit(jax.vmap(_equity_single, in_axes=(0, 0)))


def _canonicalize_pair(h: np.ndarray, v: np.ndarray) -> tuple[int, int, int, int]:
    """Canonicalize a (hero, vill) pair under suit relabeling.

    Both hero and vill are pre-sorted ascending. Returns 4-tuple of canonical
    card ids. Two pairs share a canonical iff they're equivalent under a global
    suit permutation.
    """
    cards = (int(h[0]), int(h[1]), int(v[0]), int(v[1]))
    seen = {}
    rel = []
    for c in cards:
        s = c & 3
        if s not in seen:
            seen[s] = len(seen)
        rel.append(seen[s])
    new = tuple((cards[i] >> 2) * 4 + rel[i] for i in range(4))
    # Re-sort hero and vill internally to keep canonical sortedness
    h0, h1 = new[0], new[1]
    v0, v1 = new[2], new[3]
    if h0 > h1:
        h0, h1 = h1, h0
    if v0 > v1:
        v0, v1 = v1, v0
    return (h0, h1, v0, v1)


def _enumerate_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all (hero, vill) pairs with no shared cards.

    Returns
    -------
    hero_holes : (N, 2) int32  — sorted hole cards for hero
    vill_holes : (N, 2) int32  — sorted hole cards for vill
    hero_codes : (N,) int32    — hero_a * 52 + hero_b
    vill_codes : (N,) int32    — vill_a * 52 + vill_b

    The output is sorted by hero_code * 2704 + vill_code.
    """
    hands = np.array(list(combinations(range(52), 2)), dtype=np.int32)  # (1326, 2)
    n_hands = hands.shape[0]
    hand_codes = hands[:, 0] * 52 + hands[:, 1]  # (1326,)

    # Cartesian product (hero, vill)
    hero_idx = np.repeat(np.arange(n_hands), n_hands)
    vill_idx = np.tile(np.arange(n_hands), n_hands)

    hero_pairs = hands[hero_idx]  # (N, 2)
    vill_pairs = hands[vill_idx]

    # Filter out overlapping pairs
    overlap = (
        (hero_pairs[:, 0] == vill_pairs[:, 0])
        | (hero_pairs[:, 0] == vill_pairs[:, 1])
        | (hero_pairs[:, 1] == vill_pairs[:, 0])
        | (hero_pairs[:, 1] == vill_pairs[:, 1])
    )
    keep = ~overlap
    hero_pairs = hero_pairs[keep]
    vill_pairs = vill_pairs[keep]
    hero_codes = hand_codes[hero_idx][keep]
    vill_codes = hand_codes[vill_idx][keep]

    # Sort by combined key
    combined = hero_codes.astype(np.int64) * 2704 + vill_codes.astype(np.int64)
    order = np.argsort(combined, kind="stable")
    return (
        hero_pairs[order],
        vill_pairs[order],
        hero_codes[order],
        vill_codes[order],
    )


def _build_canonical_index(
    hero_holes: np.ndarray, vill_holes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group pairs by canonical form under suit relabeling.

    Returns
    -------
    canon_heroes : (M, 2) int32  — one representative hero hand per canonical class
    canon_vills  : (M, 2) int32  — corresponding vill hand
    pair_to_canon: (N,) int32    — index into canon_* for each input pair
    """
    n = hero_holes.shape[0]
    canon_to_idx: dict[tuple[int, int, int, int], int] = {}
    pair_to_canon = np.zeros(n, dtype=np.int32)
    canon_h_list: list[tuple[int, int]] = []
    canon_v_list: list[tuple[int, int]] = []
    for i in range(n):
        key = _canonicalize_pair(hero_holes[i], vill_holes[i])
        idx = canon_to_idx.get(key)
        if idx is None:
            idx = len(canon_to_idx)
            canon_to_idx[key] = idx
            canon_h_list.append((key[0], key[1]))
            canon_v_list.append((key[2], key[3]))
        pair_to_canon[i] = idx
    canon_heroes = np.asarray(canon_h_list, dtype=np.int32)
    canon_vills = np.asarray(canon_v_list, dtype=np.int32)
    return canon_heroes, canon_vills, pair_to_canon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="src/jaxpot/env/poker/preflop_ev.npz",
        help="Output path for the npz file.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Number of (hero, vill) pairs computed per JIT call. "
        "Higher uses more memory but reduces overhead.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only compute the first N pairs (for testing).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20000,
        help="Save partial progress every N pairs (0 disables).",
    )
    args = parser.parse_args()

    print("Backend:", jax.default_backend())
    print(f"Devices: {jax.devices()}")

    print("Enumerating (hero, vill) pairs...")
    hero_holes, vill_holes, hero_codes, vill_codes = _enumerate_pairs()
    n_total = hero_holes.shape[0]
    if args.limit > 0:
        n_total = min(n_total, args.limit)
        hero_holes = hero_holes[:n_total]
        vill_holes = vill_holes[:n_total]
        hero_codes = hero_codes[:n_total]
        vill_codes = vill_codes[:n_total]
    print(f"Total pairs: {n_total:,}")
    print(f"Boards per pair: {NUM_BOARDS:,}")

    print("Building canonical-form index (suit relabeling)...")
    t0 = time.time()
    canon_heroes, canon_vills, pair_to_canon = _build_canonical_index(hero_holes, vill_holes)
    n_canon = canon_heroes.shape[0]
    print(
        f"  Canonical pairs: {n_canon:,}  "
        f"(reduction {n_total / max(n_canon, 1):.1f}x, took {time.time() - t0:.1f}s)"
    )
    print(f"Total comparisons: {n_canon * NUM_BOARDS:,}")

    canon_win = np.zeros(n_canon, dtype=np.float32)
    canon_tie = np.zeros(n_canon, dtype=np.float32)

    # Warmup JIT
    print("Warming up JIT...")
    t0 = time.time()
    h_warm = jnp.asarray(canon_heroes[: args.batch])
    v_warm = jnp.asarray(canon_vills[: args.batch])
    w, t = _equity_batch_jit(h_warm, v_warm)
    w.block_until_ready()
    print(f"Warmup took {time.time() - t0:.1f}s. First batch: win={float(w[0]):.4f} tie={float(t[0]):.4f}")

    print(f"Computing equity for {n_canon:,} canonical pairs (batch={args.batch})...")
    t_start = time.time()
    last_log = t_start
    for start in range(0, n_canon, args.batch):
        end = min(start + args.batch, n_canon)
        h = jnp.asarray(canon_heroes[start:end])
        v = jnp.asarray(canon_vills[start:end])
        # Pad to fixed batch size if needed (avoid recompilation)
        actual = end - start
        if actual < args.batch:
            pad = args.batch - actual
            h = jnp.concatenate([h, jnp.zeros((pad, 2), dtype=h.dtype)], axis=0)
            v = jnp.concatenate(
                [v, jnp.array([[2, 3]] * pad, dtype=v.dtype)], axis=0
            )
        w, t = _equity_batch_jit(h, v)
        w_np = np.asarray(w[:actual])
        t_np = np.asarray(t[:actual])
        canon_win[start:end] = w_np
        canon_tie[start:end] = t_np

        now = time.time()
        if now - last_log >= 5.0:
            elapsed = now - t_start
            done = end
            rate = done / max(elapsed, 1e-6)
            eta = (n_canon - done) / max(rate, 1e-6)
            print(
                f"  {done:>9,}/{n_canon:,}  "
                f"{100 * done / n_canon:5.1f}%  "
                f"rate={rate:7.1f} pair/s  "
                f"elapsed={elapsed/60:6.1f}m  eta={eta/60:6.1f}m"
            )
            last_log = now

        if args.checkpoint_every and end % args.checkpoint_every < args.batch:
            ckpt_path = args.out + ".partial"
            np.savez_compressed(
                ckpt_path,
                canon_win=canon_win[:end],
                canon_tie=canon_tie[:end],
                pair_to_canon=pair_to_canon,
                hero_code=hero_codes,
                vill_code=vill_codes,
                progress=np.array([end], dtype=np.int32),
            )

    elapsed = time.time() - t_start
    print(f"Equity computation done in {elapsed/60:.1f} minutes.")

    # Expand canonical results to all pairs
    print("Expanding canonical equity to all pairs...")
    win_arr = canon_win[pair_to_canon]
    tie_arr = canon_tie[pair_to_canon]

    # Sanity check
    print("Sanity: mean win =", float(win_arr.mean()))
    print("Sanity: mean tie =", float(tie_arr.mean()))

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        offsets=np.zeros(1, dtype=np.int32),
        counts=np.zeros(1, dtype=np.int32),
        hero_code=hero_codes,
        vill_code=vill_codes,
        win=win_arr,
        tie=tie_arr,
    )
    print(f"Saved to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    # Clean up partial checkpoint if exists
    partial = out_path + ".partial.npz"
    if os.path.exists(partial):
        os.remove(partial)


if __name__ == "__main__":
    main()
