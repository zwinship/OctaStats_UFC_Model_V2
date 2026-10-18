#!/usr/bin/env python3
"""
Temporary end-to-end dry run against live sources (round 4).
Runs the real pipeline code with every GitHub write replaced by a stub that
saves locally. The workflow's token is read-only, so nothing can be pushed.
"""
import importlib.util
import json
import os
import pickle
import sys
import traceback

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "diag_out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def stub_write(obj, path, *a, **k):
    dst = os.path.join(OUT, path.replace("/", "__"))
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(dst, index=False)
    elif path.endswith(".json"):
        open(dst, "w").write(json.dumps(obj, indent=2, default=str))
    else:
        pickle.dump(obj, open(dst, "wb"))
    print(f"  [dry-run: not written to GitHub] {path}")
    return True


def run(title, fn):
    section(title)
    try:
        fn()
    except SystemExit as e:
        print(f"  exit({e.code})")
    except Exception:
        traceback.print_exc()


# ── --test modes ─────────────────────────────────────────────────────────────
sys.argv = ["x", "--test"]
run("02_scrape_weekly_update --test", lambda: load("02_scrape_weekly_update.py", "m02").run_test())
run("05_scrape_results --test", lambda: load("05_scrape_results.py", "m05").run_test())
run("v1_scrape_results --test", lambda: load("v1_scrape_results.py", "mv1r").run_test())
run("v1_predict --test", lambda: load("v1_predict.py", "mv1p").run_test())


# ── V2 predictor end-to-end on the live card ─────────────────────────────────
def v2_predict():
    m = load("04_predict.py", "m04")
    m.write_csv_to_github  = lambda df, path, msg, repo_name=None, sha=None: stub_write(df, path)
    m.write_json_to_github = lambda obj, path, msg, repo_name=None: stub_write(obj, path)
    m.main()
    for f in [f for f in os.listdir(OUT) if f.startswith("predictions__v2_betting")]:
        out = pd.read_csv(os.path.join(OUT, f))
        print(out[["name", "career_style", "style_shift_probability", "Odds", "implied_probability",
                   "predicted_probability", "betting_edge", "bet_size", "prob_ko_tko",
                   "prob_submission", "prob_decision"]].to_string())
        print("  dk_link:", out[out.bet_size > 0]["dk_link"].tolist())
    for f in [f for f in os.listdir(OUT) if f.startswith("predictions__v2_prop")]:
        p = pd.read_csv(os.path.join(OUT, f))
        print(p[p.flagged][["matchup", "prop_label", "model_prob", "market_implied_prob",
                            "market_odds_str", "market_edge", "bet_size"]].to_string())
        print(f"  props with live market odds: {p.market_implied_prob.notna().sum()}/{len(p)}")
    oh = os.path.join(OUT, "data__raw__bfo_odds_history.csv")
    if os.path.exists(oh):
        print(pd.read_csv(oh).tail(24).to_string())


run("04_predict main() — live card, live odds, real model (writes stubbed)", v2_predict)


# ── V1 retrain + predict end-to-end ──────────────────────────────────────────
def v1_end_to_end():
    t = load("v1_train_model.py", "mv1t")
    t.write_csv_to_github    = lambda df, path, msg, repo=None: stub_write(df, path)
    t.write_pickle_to_github = lambda obj, path, msg, repo=None: stub_write(obj, path)
    t.main()
    bundle = pickle.load(open(os.path.join(OUT, "data__v1_model_bundle.pkl"), "rb"))
    print("  style counts:", bundle["style_counts"])
    p = load("v1_predict.py", "mv1p2")
    p.read_pickle_from_github = lambda path, repo=None: bundle
    p.write_csv_to_github = lambda df, path, msg, repo=None: stub_write(df, path)
    p.main()
    f = os.path.join(OUT, "predictions__v1_predictions_latest.csv")
    if os.path.exists(f):
        print(pd.read_csv(f).to_string())


run("V1 retrain + predict — live card (writes stubbed)", v1_end_to_end)

print("\nDIAG COMPLETE")
