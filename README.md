# Sequence Board Game AI Agent

AI agents for the [Sequence](https://en.wikipedia.org/wiki/Sequence_(game)) board game, built with heuristic evaluation, minimax search, belief-state determinization, reinforcement learning weight tuning, and a CNN value network.

**Team:** Muskan Jain, Satyaa Sudarshan Gurswamy Sethuraman

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

---

## How to Run

### Play an interactive game
```bash
python play.py human-vs-ai       # you vs the AI
python play.py human-vs-human    # two humans at the same terminal
python play.py ai-vs-ai          # watch Combined Agent vs Greedy Agent
```

### Run tournaments
```bash
# All 5 agents vs each other (50 games each)
python -m evaluation.run_full_comparison

# Ablation study: isolate each component's contribution
python -m evaluation.run_ablation

# Baseline tournaments (Random, Greedy, Combined hand-tuned)
python -m evaluation.run_baselines
```

### Training
```bash
# RL hill-climbing weight tuning (resumes from learned_weights.json if it exists)
python -m training.run_training

# CNN value network training pipeline
python -m training.nn_pipeline
```

### Tests
```bash
python -m pytest tests/ -v
```

### Utilities
```bash
# Validate the board layout
python -c "from game.board import validate_board_layout; validate_board_layout()"

# Check learned RL weights
python -c "import json; d=json.load(open('training/learned_weights.json')); print(d['weights'])"
```

---

## Agents

| Agent | Description |
|-------|-------------|
| `RandomAgent` | Picks a uniformly random legal move |
| `GreedyAgent` | Picks the move with the best immediate heuristic score |
| `CombinedAgent` (hand-tuned) | Minimax search + belief-state determinization, default weights |
| `CombinedAgent` (RL-tuned) | Same architecture, weights optimized by hill-climbing self-play |
| `NNAgent` | Minimax search + belief-state, evaluated by a CNN value network |

---

## Project Structure

```
shared/         — Card, Move, GameState types and constants
game/           — Board layout, deck, move generation, win detection, game loop
agent/          — Heuristic evaluator, minimax search, belief model, all agents
training/       — RL weight tuning, CNN training pipeline, trained models
evaluation/     — Tournament runner, baseline comparisons, ablation studies
tests/          — Unit and integration tests
play.py         — Interactive game interface
main.py         — Entry point
```

---

## Key Results (50-game tournaments)

| Matchup | Agent 1 Win% | Agent 2 Win% |
|---------|-------------|-------------|
| Random vs Greedy | 0% | 100% |
| Random vs Combined (hand-tuned) | 0% | 100% |
| Greedy vs Combined (hand-tuned) | 42% | 58% |
| Combined (hand-tuned) vs Combined (RL-tuned) | 44% | 56% |
