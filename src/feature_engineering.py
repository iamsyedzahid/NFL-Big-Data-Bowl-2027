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

def extract_game_kinematics() -> pl.DataFrame:
    """Extracts in-game speed, acceleration, and movement metrics across all available game tracking files."""
    game_files = [
        os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
        if f.startswith("game_tracking_") and f.endswith(".csv")
    ]
    if not game_files:
        return pl.DataFrame()

    dfs = []
    for filepath in game_files:
        df = pl.scan_csv(filepath, null_values=["NA", "N/A", ""])
        # Aggregate tracking metrics per player
        player_metrics = (
            df.filter(pl.col("nfl_id").is_not_null())
            .group_by("nfl_id")
            .agg([
                pl.col("s").max().alias("game_max_speed"),
                pl.col("s").mean().alias("game_avg_speed"),
                pl.col("a").max().alias("game_max_accel"),
            ])
            .collect()
        )
        dfs.append(player_metrics)

    if not dfs:
        return pl.DataFrame()

    # Combine metrics across all seasons
    combined = pl.concat(dfs).group_by("nfl_id").agg([
        pl.col("game_max_speed").max(),
        pl.col("game_avg_speed").mean(),
        pl.col("game_max_accel").max()
    ])
    return combined

def build_features() -> pl.DataFrame:
    players_path = os.path.join(DATA_DIR, "players.csv")
    combine_path = os.path.join(DATA_DIR, "combine_results.csv")
    success_path = os.path.join(DATA_DIR, "player_career_successes.csv")

    if not (os.path.exists(players_path) and os.path.exists(combine_path) and os.path.exists(success_path)):
        return pl.DataFrame()

    # Read base CSVs with schema overrides for missing/string values
    players = pl.read_csv(
        players_path,
        null_values=["NA", "N/A", ""],
        schema_overrides={"draft_round": pl.Utf8}
    )
    combine = pl.read_csv(
        combine_path,
        null_values=["NA", "N/A", ""]
    )
    targets = pl.read_csv(
        success_path,
        null_values=["NA", "N/A", ""]
    )

    # Cast nfl_id to Int64 across dataframes to ensure smooth joins
    players = players.with_columns(pl.col("nfl_id").cast(pl.Int64))
    combine = combine.with_columns(pl.col("nfl_id").cast(pl.Int64))
    targets = targets.with_columns(pl.col("nfl_id").cast(pl.Int64))

    # Binary Target: High Impact Player
    targets = targets.with_columns(
        ((pl.col("career_games_started") / (pl.col("career_games_active") + 1e-5) > 0.5) |
         (pl.col("pro_bowl_original_ballot") > 0) |
         (pl.col("ap_all_pro_1st_team") > 0)).cast(pl.Int32).alias("target_high_impact")
    )

    # Join base data
    dataset = players.join(combine, on="nfl_id", how="left").join(targets, on="nfl_id", how="left")

    # Join Combine Tracking
    combine_kinematics = extract_combine_kinematics()
    if len(combine_kinematics) > 0:
        combine_kinematics = combine_kinematics.with_columns(pl.col("nfl_id").cast(pl.Int64))
        dataset = dataset.join(combine_kinematics, on="nfl_id", how="left")

    # Join In-Game Tracking
    game_kinematics = extract_game_kinematics()
    if len(game_kinematics) > 0:
        game_kinematics = game_kinematics.with_columns(pl.col("nfl_id").cast(pl.Int64))
        dataset = dataset.join(game_kinematics, on="nfl_id", how="left")

    return dataset