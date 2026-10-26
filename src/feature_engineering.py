import os
import polars as pl
import numpy as np

DATA_DIR = "data"

def extract_combine_kinematics() -> pl.DataFrame:
    """Extracts burst, peak acceleration, and motion efficiency from Combine optical tracking."""
    tracking_path = os.path.join(DATA_DIR, "combine_tracking.csv")
    if not os.path.exists(tracking_path):
        return pl.DataFrame()

    df = pl.scan_csv(tracking_path)
    features = (
        df.filter(pl.col("entity_type") == "PLAYER")
        .group_by("nfl_id")
        .agg([
            pl.col("s").max().alias("combine_max_speed"),
            pl.col("a").max().alias("combine_max_accel"),
            (pl.col("a") / (pl.col("s") + 1e-5)).mean().alias("avg_jerk_ratio")
        ])
        .collect()
    )
    return features

def build_features() -> pl.DataFrame:
    players_path = os.path.join(DATA_DIR, "players.csv")
    combine_path = os.path.join(DATA_DIR, "combine_results.csv")
    success_path = os.path.join(DATA_DIR, "player_career_successes.csv")

    if not (os.path.exists(players_path) and os.path.exists(combine_path) and os.path.exists(success_path)):
        return pl.DataFrame()

    players = pl.read_csv(players_path)
    combine = pl.read_csv(combine_path)
    targets = pl.read_csv(success_path)

    targets = targets.with_columns(
        ((pl.col("career_games_started") / (pl.col("career_games_active") + 1e-5) > 0.5) |
         (pl.col("pro_bowl_original_ballot") > 0) |
         (pl.col("ap_all_pro_1st_team") > 0)).cast(pl.Int32).alias("target_high_impact")
    )

    dataset = players.join(combine, on="nfl_id", how="left").join(targets, on="nfl_id", how="left")
    kinematics = extract_combine_kinematics()

    if len(kinematics) > 0:
        dataset = dataset.join(kinematics, on="nfl_id", how="left")

    return dataset