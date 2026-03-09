# OctaStats UFC Model V2

[![Wednesday Update](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/wednesday_data_update.yml/badge.svg)](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/wednesday_data_update.yml)
[![Friday Predictions](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/friday_predictions.yml/badge.svg)](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/friday_predictions.yml)
[![Sunday Results](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/sunday_results.yml/badge.svg)](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/sunday_results.yml)

Data-driven UFC fight prediction model using a **Discrete Choice Dynamic Logit** with LASSO feature selection and bounded Kelly bet sizing. Fully automated via GitHub Actions.

**Live site:** [OctaStats.github.io](https://OctaStats.github.io)

---

## Model Overview

### Architecture

| Component | Detail |
|---|---|
| Model type | Discrete Choice Dynamic Logit |
| Feature selection | LASSO-penalised Logistic Regression (L1, cross-validated C) |
| State variables | Fighter's last 3 in-fight styles (captures style momentum) |
| Bet sizing | Bounded Kelly criterion via logistic transform (max ≈ 5u, never reached) |
| Odds source | BestFightOdds (DraftKings preferred, FanDuel fallback) |
| Data source | UFCStats.com (2015–present) |
| Retrain schedule | Weekly (Wednesday) |

### Fighting Style Categories

Six styles classified from career-average stats, standardised within weight class:

- **Striker** — High KD, sig strikes, total strikes, head attacks
- **Wrestler** — High control time, ground attacks, takedowns
- **BJJ** — High submission attempts + control time + takedowns
- **Muay Thai** — High clinch, body, leg, and head attacks
- **Sniper** — High distance strikes with elevated accuracy
- **Mixed** — Doesn't meet any dominant threshold

> **Key constraint:** A fighter's style label for a historical fight is **locked** — it reflects their stats going into that fight and never changes when new data is added. Only the predicted in-fight style for upcoming fights is forward-looking.

### Dynamic Component

The model includes lag state variables: each fighter's last 1, 2, and 3 in-fight styles. This allows the model to detect style momentum (e.g. a wrestler who has been fighting like a striker lately) and flag potential style shifts (shown on the website when model confidence ≥ 78%).

### Bet Sizing Formula

```
kelly_fraction = (p × b - q) / b
confidence     = |p - 0.5| × 2          # how far from coin-flip
adj_steepness  = STEEPNESS × confidence  # STEEPNESS = 3.5
units          = 5.0 × sigmoid(adj_steepness × kelly_fraction)
```

This logistic transformation ensures units approach but **never reach** 5.00. A confident 40% edge produces ~2.5u; a narrow 5% edge produces ~0.3u.

---

## Repository Structure

```
OctaStats_UFC_Model_V2/
├── .github/
│   └── workflows/
│       ├── wednesday_data_update.yml   # Scrape + retrain
│       ├── friday_predictions.yml      # Predict (if fight week)
│       └── sunday_results.yml          # Results + stats
│
├── scripts/
│   ├── 01_scrape_historical.py         # One-time historical scrape (run locally)
│   ├── 02_scrape_weekly_update.py      # Incremental weekly scrape
│   ├── 03_train_model.py               # Feature engineering + model training
│   ├── 04_predict.py                   # Generate weekly predictions
│   ├── 05_scrape_results.py            # Record post-fight results
│   └── 06_statistics.py               # Statistical analysis
│
├── data/
│   ├── raw/
│   │   └── fight_stats.csv             # Per-fight scraped data (2015-present)
│   ├── processed/
│   │   └── career_styled.csv           # Career stats + style labels (leak-free)
│   └── model_bundle.pkl                # Serialised model + metadata
│
├── predictions/
│   └── v2_betting_recommendations_{event}.csv
│
├── results/
│   ├── v2_betting_results_{event}.csv
│   └── v2_all_betting_results.csv
│
├── statistics/
│   └── v2_statistical_analysis.json
│
└── titles/
    └── fight_titles.json               # upcoming + recent event names
```

---

## Automation Schedule

| Day | Time | Action |
|---|---|---|
| Wednesday | 12:00 UTC | Scrape new fight data → retrain model → notify site |
| Friday | 12:00 UTC | Check for fight this week → if yes, run predictions → notify site |
| Sunday | 12:00 UTC | Scrape fight results → compute P&L → run stats → notify site |

All workflows can also be triggered manually from the GitHub Actions tab.

---

## Setup Instructions

See `SETUP.md` for the complete step-by-step setup guide.

---

## V1 vs V2 Comparison

| Feature | V1 | V2 |
|---|---|---|
| Model | Linear Regression (per weight class + style) | Dynamic Logit + LASSO |
| Data | Static CSV files | Live-scraped, weekly updated |
| Bet sizing | Discrete (0, 1, 2, 3 units) | Continuous (0.00–4.99 units) |
| Style dynamics | None | 3-lag style state variables |
| Style shift prediction | No | Yes (≥78% confidence threshold) |
| Odds source | fightodds.io (Selenium) | BestFightOdds (simple HTTP) |
| Automation | Manual Google Colab runs | Fully automated GitHub Actions |
| Website | PythonAnywhere (Flask) | GitHub Pages (static) |

---

## Data Notes

- All career stats are computed **strictly before** each fight date — no data leakage.
- Women's weight classes are included in training but the model trains separate style classifiers per weight class.
- Fights where both fighters have fewer than 3 prior UFC fights are excluded from model training (too noisy).
- Stats from UFCStats are sometimes adjusted 1–2 days after events — the Wednesday scrape is intentionally delayed to capture final values.
