# OctaStats UFC Model V2

[![Wednesday Update](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/wednesday_data_update.yml/badge.svg)](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/wednesday_data_update.yml)
[![Friday Predictions](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/friday_predictions.yml/badge.svg)](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/friday_predictions.yml)
[![Sunday Results](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/sunday_results.yml/badge.svg)](https://github.com/zwinship/OctaStats_UFC_Model_V2/actions/workflows/sunday_results.yml)

**Live site:** [OctaStats.github.io](https://OctaStats.github.io)

---

## Background

About a year ago I built my first UFC prediction model — a linear regression approach that grouped fighters by weight class and fighting style, then predicted outcomes based on statistical differentials. It worked, I learned a lot, and I kept running it manually out of a Google Colab notebook every fight week.

A year on, I have spent a lot of time studying data science more seriously, and I wanted to see how much better I could make it. V2 is the result — a ground-up rebuild that keeps the core idea I was most proud of from V1 (fighting style classification) but layers real statistical depth on top of it, and automates the entire pipeline so it runs itself every week without me touching it.

The two things I think make this model genuinely interesting are the **fighting style system** and the **dynamic logit architecture** — both described below.

---

## What Makes This Model Different

### Fighting Style Classification

The foundation of this model — carried over from V1 and refined — is the idea that raw stats alone don't tell the full story of a fight. What matters is *how* a fighter fights, and whether their style creates problems for their opponent's style.

Every fighter is assigned one of six styles based on their career-average statistics, standardised within their weight class so a "high takedown rate" means something different at Flyweight than at Heavyweight:

| Style | Signature |
|---|---|
| **Striker** | High knockdowns, significant strikes, head attack volume |
| **Wrestler** | High control time, takedown volume, ground strikes |
| **BJJ** | High submission attempts combined with control time and takedowns |
| **Muay Thai** | High clinch work, body and leg attack volume |
| **Sniper** | High distance striking accuracy with low takedown engagement |
| **Mixed** | No single dominant dimension — generalist profile |

The key design constraint: a fighter's style label for any historical fight is **permanently locked** at the stats they had going *into* that fight. It never gets retroactively updated as new data comes in. This prevents data leakage and means the model is always learning from what was actually knowable at the time.

Style matchups — Striker vs Wrestler, BJJ vs Muay Thai, Sniper vs Mixed — are encoded directly into the model's feature matrix, so it learns which stylistic matchups historically favour which fighter type.

### Dynamic Logit with Style State Variables

Where V2 goes significantly further than V1 is in treating fighting style as something that *evolves* over time rather than being static. A fighter might be classified as a Wrestler by career average, but if their last three fights they have been landing like a Striker, that trend matters.

The model captures this through **lag state variables** — each fighter's in-fight style from their last 1, 2, and 3 fights is encoded and fed into the model alongside their career averages. This gives the logit a dynamic dimension: it is not just asking "what kind of fighter is this?" but "what kind of fighter have they been acting like lately, and is that trending away from their baseline?"

This momentum signal feeds two outputs:

**Win probability** — the core prediction, where style momentum is one of the inputs the LASSO-penalised logit weighs against all other career stat differentials.

**Style shift prediction** — a secondary model that specifically predicts whether a fighter is likely to deviate from their career style in this fight. When confidence exceeds 78%, this is flagged on the website as a warning signal — because a fighter about to fight outside their comfort zone is one of the harder-to-quantify edges the model is designed to surface.

These are exactly the kinds of patterns that do not show up in a simple regression on raw stats. Two fighters can have near-identical career averages and the style momentum signal alone can swing the predicted probability meaningfully.

---

## Model Architecture

| Component | Detail |
|---|---|
| Model type | Discrete Choice Dynamic Logit |
| Feature selection | LASSO-penalised Logistic Regression (L1, cross-validated C) |
| Style classification | 6 categories, z-scored within weight class, historically locked |
| State variables | Fighter's last 3 in-fight styles (style momentum lags) |
| Style shift detection | Binary logit, displayed on site at ≥78% confidence |
| Bet sizing | Bounded Kelly criterion via logistic transform (max ≈ 5u, never reached) |
| Odds source | BestFightOdds (DraftKings preferred, FanDuel fallback) |
| Data source | UFCStats.com (2015–present) |
| Retrain schedule | Every Wednesday, fully automated |

### Bet Sizing

```
kelly_fraction = (p × b - q) / b
confidence     = |p - 0.5| × 2
adj_steepness  = STEEPNESS × confidence    # STEEPNESS = 3.5
units          = 5.0 × sigmoid(adj_steepness × kelly_fraction)
```

Bet sizes are continuous to 2 decimal places and approach but never reach 5.00u. A confident 40% edge produces ~2.5u; a narrow 5% edge produces ~0.3u. Minimum edge of 4 percentage points required to place any bet.

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

## V1 vs V2

| | V1 | V2 |
|---|---|---|
| Model | Linear Regression (per weight class + style group) | Dynamic Logit + LASSO |
| Style dynamics | Static career average only | 3-lag style state variables |
| Style shift detection | No | Yes (≥78% confidence) |
| Data | Static CSV files | Live-scraped, weekly updated |
| Bet sizing | Discrete (0, 1, 2, 3u) | Continuous (0.00–4.99u) |
| Odds source | fightodds.io (Selenium) | BestFightOdds (HTTP) |
| Automation | Manual Colab runs | Fully automated GitHub Actions |
| Website | PythonAnywhere Flask | GitHub Pages (static) |

---

## Data Notes

- All career stats are computed strictly before each fight date — no data leakage.
- Style labels for historical fights are permanently locked at the time of that fight.
- Women's weight classes are included; style classification is computed separately within each weight class.
- Fights where both fighters have fewer than 3 prior UFC fights are excluded from training.
- UFCStats sometimes adjusts stats 1–2 days post-event — the Wednesday scrape is intentionally scheduled to capture final values.

See `SETUP.md` for installation and configuration instructions.
