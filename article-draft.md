# We Built a Poker Bot and Open-Sourced as a Framework

*We got hired to build a poker bot.*

*To get the desired win rate we needed self-play (PPO and AlphaZero style), league training, imperfect-information environments, and the ability to iterate fast without rewriting the training loop.*

*So we built all of that. Now we’re opensourcing the infrastructure as its own library: [Jaxpot](https://github.com/bards-ai/Jaxpot#). This post walks you through it with Colab notebook .*



## Open The Notebook

We’ll start with simple self-play agent. It takes about 90 seconds on a free Colab GPU and proves the whole pipeline works. After that, you’ll switch config and train on Dark Hex, a game where you can’t see your opponent’s moves.

Here’s the notebook:

```text
https://colab.research.google.com/drive/1-rm_Bh8CNaM861We97ZoicfgKxz0xOSi?usp=sharing
```

TODO: In the published repo this notebook should be linked near the top of the

Open it, hit “Run all”. We recommend loging it with Weights & Biases account to see the training artifacts. If you don’t have the account you can preview them in TensorBoard.



## Reading The Training Output

When the run starts, Jaxpot prints a TUI dashboard.

The dashboard is useful while the run is active, but curves are better for understanding training. You should see something like this after the run:

The example used tic tac toe as learning environment so you can see model absolutely crushing it right from the start. You can start the training for Dark Hex from the same notebook and read the rest of article while it’s training :D

## [Jaxpot](https://github.com/bards-ai/Jaxpot#)

It is built around three practical ideas:

- **PPO and AlphaZero-style training.** PPO gives you a strong policy-gradient baseline for self-play. AlphaZero-style components are useful when you want search and value-guided planning.
- **JAX.**  Vectorized rollouts and training, compiled, and run efficiently on accelerators. Pushing the training speed to the hardware limit.
- **Hydra configs.** Experiments are composed from small config files for the game, model, trainer, evaluator, and logger. Changing the game or training setup requires just changing the config.

## Why It Is A Good Fit For Imperfect-Information Games

In perfect information games like Chess, Go, and standard Hex, both players see the full board. The game state is public.

Imperfect-information games split the state into public and private parts. Poker hides cards. Liar’s Dice hides dice. Dark Hex hides opponent stones unless you collide with them.

That changes everything. The optimal strategy is not a single fixed policy. A poker bot that always plays the highest win-rate line becomes predictable and easy to exploit.

Self-play already helps: the opponent keeps changing, so the policy cannot overfit to a fixed strategy. Entropy scheduling keeps the policy exploring early in training, which matters when the game demands mixed strategies.

On top of that, Jaxpot supports League play. The agent trains against frozen snapshots of itself from earlier in the run, and opponents it struggles against get higher sampling weight. That prevents the policy from “forgetting” how to beat older versions of itself. When the league fills up, surplus opponents move to an archive that can reactivate if the agent starts losing to them again.



## Train On Harder Game: Dark Hex

Now let’s switch to a game that is still small enough to understand, but much more interesting.

Dark Hex is an imperfect-information version of [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)).

In normal Hex, both players see the whole board. In Dark Hex, each player sees:

- empty cells according to their own view
- their own stones
- opponent stones only when revealed through failed placement attempts

The true board exists, but the agent does not get to see it.

This is a much better demonstration of why environment design matters. The agent is not just choosing a move from a board. It is acting under uncertainty.

Jaxpot already includes the Dark Hex environment:

Just head back to the notebook and see if you can improve the Dark Hex Agent winrate by changing the parameters!

[Run in Colab on free GPU](https://colab.research.google.com/drive/1-rm_Bh8CNaM861We97ZoicfgKxz0xOSi?usp=sharing)