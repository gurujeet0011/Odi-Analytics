"""
ODI Cricket Analytics Dashboard
--------------------------------
A Flask web app for exploring ODI cricket team & player statistics and
ML-predicted future performance (RandomForest models trained on historical
season-to-season and match-to-match data).

Run:
    
    python app.pypip install -r requirements.txt
Then open:
    http://127.0.0.1:5000
"""
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load data & models once at startup
# ---------------------------------------------------------------------------
team_overall = pd.read_csv(os.path.join(DATA_DIR, "team_overall_stats.csv"))
team_season = pd.read_csv(os.path.join(DATA_DIR, "team_season_stats.csv"))
team_form = pd.read_csv(os.path.join(DATA_DIR, "team_current_form.csv"))
team_pred = pd.read_csv(os.path.join(DATA_DIR, "team_predictions.csv"))

player_season = pd.read_csv(os.path.join(DATA_DIR, "player_season_stats.csv"))
player_pred = pd.read_csv(os.path.join(DATA_DIR, "player_predictions.csv"))
batters = pd.read_csv(os.path.join(DATA_DIR, "batter_player_stats.csv"))
bowlers = pd.read_csv(os.path.join(DATA_DIR, "bowler_player_stats.csv"))

team_bundle = joblib.load(os.path.join(MODEL_DIR, "team_model.pkl"))
TEAM_CLF, TEAM_FEATS = team_bundle["model"], team_bundle["features"]

player_bundle = joblib.load(os.path.join(MODEL_DIR, "player_models.pkl"))
M_RUNS, M_WKTS, M_FP = player_bundle["model_runs"], player_bundle["model_wkts"], player_bundle["model_fp"]
PLAYER_FEATS = player_bundle["features"]

ALL_TEAMS = sorted(team_overall["team"].unique().tolist())
ALL_PLAYERS = sorted(player_season["player"].unique().tolist())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def team_prediction(team_name):
    row = team_form[team_form["team"] == team_name]
    if row.empty:
        return None
    r = row.iloc[0]
    feat_map = {
        "roll_winrate_10": r["winrate_last10"],
        "roll_winrate_5": r["winrate_last5"],
        "career_matches_so_far": r["matches_played"],
        "roll_runs_10": r["avg_runs_last10"],
        "roll_wkts_10": r["avg_wkts_last10"],
    }
    X = pd.DataFrame([[feat_map[f] for f in TEAM_FEATS]], columns=TEAM_FEATS)
    prob = float(TEAM_CLF.predict_proba(X)[:, 1][0])
    return round(prob, 3)


def player_prediction(player_name):
    g = player_season[player_season["player"] == player_name].sort_values("season")
    if g.empty:
        return None
    latest = g.iloc[-1]
    season_idx = len(g) - 1
    feat_map = {
        "cur_matches": latest["matches"],
        "cur_runs": latest["runs"],
        "cur_sr": latest["batting_sr"],
        "cur_wickets": latest["wickets"],
        "cur_econ": latest["bowling_econ"],
        "cur_catches": latest["catches"],
        "cur_fantasy": latest["fantasy_points"],
        "career_season_no": season_idx,
    }
    X = pd.DataFrame([[feat_map[f] for f in PLAYER_FEATS]], columns=PLAYER_FEATS)
    pred_runs = float(M_RUNS.predict(X)[0])
    pred_wkts = float(M_WKTS.predict(X)[0])
    pred_fp = float(M_FP.predict(X)[0])
    return {
        "predicted_next_runs": round(pred_runs, 1),
        "predicted_next_wickets": round(pred_wkts, 1),
        "predicted_next_fantasy_points": round(pred_fp, 1),
    }


# ---------------------------------------------------------------------------
# Routes - pages
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    top_teams = team_overall.sort_values("win_rate", ascending=False).head(8).to_dict("records")
    top_batters = batters.sort_values("total_runs", ascending=False).head(8).to_dict("records")
    top_bowlers = bowlers.sort_values("total_wickets_taken", ascending=False).head(8).to_dict("records")
    top_predicted = player_pred.sort_values("predicted_next_fantasy_points", ascending=False).head(8).to_dict("records")
    top_win_prob = team_pred.sort_values("predicted_win_prob", ascending=False).head(8).to_dict("records")
    stats = {
        "total_matches": int(team_overall["matches"].sum() // 2) if False else int(pd.read_csv(os.path.join(DATA_DIR, "team_season_stats.csv"))["matches"].sum() // 2),
        "total_teams": len(ALL_TEAMS),
        "total_players": len(ALL_PLAYERS),
    }
    return render_template(
        "dashboard.html",
        teams=ALL_TEAMS,
        players=ALL_PLAYERS,
        top_teams=top_teams,
        top_batters=top_batters,
        top_bowlers=top_bowlers,
        top_predicted=top_predicted,
        top_win_prob=top_win_prob,
        stats=stats,
    )


@app.route("/team/<team_name>")
def team_detail(team_name):
    if team_name not in ALL_TEAMS:
        return "Team not found", 404
    overall = team_overall[team_overall["team"] == team_name].iloc[0].to_dict()
    season = team_season[team_season["team"] == team_name].sort_values("season")
    pred_prob = team_prediction(team_name)
    top_players = (
        player_season[player_season["team"] == team_name]
        .groupby("player", as_index=False)
        .agg(runs=("runs", "sum"), wickets=("wickets", "sum"), fantasy_points=("fantasy_points", "sum"))
        .sort_values("fantasy_points", ascending=False)
        .head(10)
    )
    return render_template(
        "team.html",
        team=team_name,
        overall=overall,
        seasons=season.to_dict("records"),
        pred_prob=pred_prob,
        top_players=top_players.to_dict("records"),
        teams=ALL_TEAMS,
        players=ALL_PLAYERS,
    )


@app.route("/player/<player_name>")
def player_detail(player_name):
    if player_name not in ALL_PLAYERS:
        return "Player not found", 404
    g = player_season[player_season["player"] == player_name].sort_values("season")
    prediction = player_prediction(player_name)
    bat_row = batters[batters["player_name"] == player_name]
    bowl_row = bowlers[bowlers["player_name"] == player_name]
    career = {}
    if not bat_row.empty:
        career.update(bat_row.iloc[0].to_dict())
    if not bowl_row.empty:
        for k, v in bowl_row.iloc[0].to_dict().items():
            career.setdefault(k, v)
    return render_template(
        "player.html",
        player=player_name,
        seasons=g.to_dict("records"),
        prediction=prediction,
        career=career,
        teams=ALL_TEAMS,
        players=ALL_PLAYERS,
    )


# ---------------------------------------------------------------------------
# Routes - JSON API (used by charts / search)
# ---------------------------------------------------------------------------
@app.route("/api/team/<team_name>")
def api_team(team_name):
    if team_name not in ALL_TEAMS:
        return jsonify({"error": "not found"}), 404
    season = team_season[team_season["team"] == team_name].sort_values("season")
    return jsonify({
        "team": team_name,
        "seasons": season.to_dict("records"),
        "predicted_win_prob": team_prediction(team_name),
    })


@app.route("/api/player/<player_name>")
def api_player(player_name):
    if player_name not in ALL_PLAYERS:
        return jsonify({"error": "not found"}), 404
    g = player_season[player_season["player"] == player_name].sort_values("season")
    return jsonify({
        "player": player_name,
        "seasons": g.to_dict("records"),
        "prediction": player_prediction(player_name),
    })


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    kind = request.args.get("type", "player")
    pool = ALL_PLAYERS if kind == "player" else ALL_TEAMS
    matches = [p for p in pool if q in p.lower()][:15]
    return jsonify(matches)

import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
