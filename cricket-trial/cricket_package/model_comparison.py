"""Compare Linear Regression, Random Forest and XGBoost on dataset.csv.

Each configuration is run 25 times (seeds 0..24). Every run uses a different random
80/20 train/test split and, where applicable, the same seed for the model.
Reports mean +- SD of R2, RMSE and MAE across the 25 runs.
"""
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

N_RUNS = 25

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "dataset.csv"))
X, y = df[["x"]].values, df["y"].values

configs = {"LinearRegression": [("default", LinearRegression())]}
configs["LinearRegression"] += [(f"Ridge(alpha={a})", Ridge(alpha=a)) for a in (0.1, 1, 10)]
configs["LinearRegression"] += [(f"Lasso(alpha={a})", Lasso(alpha=a)) for a in (0.01, 0.1)]

configs["RandomForest"] = [
    (f"n={n},depth={d}", RandomForestRegressor(n_estimators=n, max_depth=d))
    for n in (50, 100, 300) for d in (None, 5, 10)
]
configs["XGBoost"] = [
    (f"n={n},lr={lr},depth={d}",
     XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=d, verbosity=0))
    for n in (100, 300) for lr in (0.05, 0.1) for d in (3, 6)
]

rows = []
for family, items in configs.items():
    for name, model in items:
        r2s, rmses, maes = [], [], []
        for seed in range(N_RUNS):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            m = clone(model)
            if "random_state" in m.get_params():
                m.set_params(random_state=seed)
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            r2s.append(r2_score(yte, p))
            rmses.append(np.sqrt(mean_squared_error(yte, p)))
            maes.append(mean_absolute_error(yte, p))
        f = lambda a: f"{np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}"
        rows.append({"Model": family, "Configuration": name, "Runs": N_RUNS,
                     "R2": f(r2s), "RMSE": f(rmses), "MAE": f(maes)})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(os.path.join(HERE, "model_comparison_results.csv"), index=False, encoding="utf-8-sig")
with open(os.path.join(HERE, "model_comparison_results.md"), "w", encoding="utf-8") as fh:
    cols = list(out.columns)
    fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in out.itertuples(index=False):
        fh.write("| " + " | ".join(str(v) for v in r) + " |\n")
