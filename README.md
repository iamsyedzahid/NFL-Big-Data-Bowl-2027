---
title: NFL Prospect Evaluator
emoji: 🏈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---

# NFL Prospect MLOps Pipeline (NFL Big Data Bowl 2027 Dataset)

This repository contains a full MLOps pipeline for processing NFL rookie prospect tracking, combine metrics, and in-game performance.

## Data Directory Setup
Place the raw competition files inside `data/`:
- `data/players.csv`
- `data/combine_results.csv`
- `data/combine_tracking.csv`
- `data/player_career_successes.csv`
- `data/player_play.csv`
- `data/games.csv`
- `data/game_tracking_2023.csv`
- `data/game_tracking_2024.csv`
- `data/game_tracking_2025.csv`

## Local Setup & Pipeline Execution

1. Install dependencies:
   ```bash
   pip install -r requirements.txt