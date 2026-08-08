# ODI Cricket Analytics Dashboard (Flask + ML)

# ODI Cricket Analytics Dashboard

**🔗 Live Demo:** https://odi-analytics.onrender.com

A Flask-based web dashboard with ML models to predict team & player 
performance from historical ODI data.


## What's inside
- `app.py` — Flask app (routes + prediction logic)
- `templates/` — dashboard, team page, player page (Jinja2 + Chart.js)
- `static/style.css` — styling
- `data/` — precomputed CSVs (team/player season stats, predictions)
- `models/`
  - `player_models.pkl` — RandomForestRegressor models (predict a player's next-season
    runs, wickets, and fantasy points from their most recent season's stats)
  - `team_model.pkl` — RandomForestClassifier (predicts a team's win probability
    for its next match from rolling form: win rate and run/wicket rate over its
    last 5–10 matches)
- `retrain.py` — regenerates the CSVs and retrains both models from the raw dataset

## Run it locally

```bash
cd flask_app
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

If you see a scikit-learn version warning, run:
```bash
pip install scikit-learn==1.8.0
```

## Pages
- `/` — dashboard: overview stats, top teams/players, top predicted performers
- `/team/<name>` — team stats, season trend chart, predicted next-match win %
- `/player/<name>` — player career stats, season trend chart, predicted next-season performance
- Search box in the header jumps straight to any team or player

## Retraining on new data
Drop updated CSVs into the original dataset folder and run:
```bash
python retrain.py
```
This regenerates everything in `data/` and `models/`.

## Notes
- Predictions are statistical estimates from historical patterns, not guarantees —
  cricket has a lot of variance match to match.
- Team model test accuracy ≈ 60%, AUC ≈ 0.62. Player models beat a
  "same as last season" baseline on held-out data (see `retrain.py` output).
