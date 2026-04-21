# Jaxpot Go evaluation

Step-by-step workflow to run a round-robin league with Gomill and rate the engines with [BayesElo](https://www.remi-coulom.fr/Bayesian-Elo/).

---

## 1. Prepare the Gomill config

Prepare `gomill_config.ctl` so that it defines, i. e.

```python
competition_type = 'allplayall'
board_size = 9
komi = 6.5

players = {
    'kata1'  : Player('path/to/katago gtp ...'),
    'kata0'  : Player('path/to/katago gtp ...'),
    'mymodel': Player('/path/to/scripts/go.sh'),
}
rounds = 2
competitors = ['kata0', 'kata1', 'jaxpot', 'pachi']
```

Adjust paths and names to your setup. GTP wrapper scripts are in `scripts/`: `scripts/go.sh` (typical policy checkpoint) and `scripts/go_alphazero.sh` (MCTS AlphaZero); edit them for your venv, checkpoint, and `scripts/gtp_engine.py` options.

---

## 2. Run the games

From the project root:

```bash
ringmaster gomill_config.ctl
```

This runs the league and writes SGFs into `gomill_config.games/` (and updates `gomill_config.status`, `gomill_config.report`, etc.).

---

## 3. Convert results to PGN for BayesElo

Convert the Gomill SGFs to a single PGN file that BayesElo can read:

```bash
python scripts/sgf_to_bayeselo.py --games-dir gomill_config.games
```

By default this produces `results.pgn`. Use `--output <file>` to change the output path.

---

## 4. Compute Elo ratings with BayesElo

Run BayesElo (your build path may differ):

```bash
bayeselo
```

In the BayesElo prompt, run:

```
ResultSet> readpgn results.pgn
ResultSet> elo
ResultSet-EloRating> mm
ResultSet-EloRating> ratings
```

- **readpgn results.pgn** — load the PGN (reports how many games were loaded).
- **elo** — switch to Elo rating mode.
- **mm** — run the algorithm (minimum margin, or use another method if you prefer).
- **ratings** — print the ranked list with Elo, uncertainty, games, score, opposition, draws.

Example output:

```
Rank Name                               Elo    +    - games score oppo. draws
   1 KataGo:1.16.4+b28c512nbt-s12192M   237  263  263     6  100%   -79    0%
   2 Pachi:12.88                         74  226  226     6   67%   -25    0%
   3 mymodel-Go:1.0                     -74  226  226     6   33%    25    0%
   4 KataGo:1.16.4+b6c96-s1248K        -237  264  264     6    0%    79    0%
```

