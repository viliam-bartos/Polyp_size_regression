import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    df_merged = pd.read_csv("polyp_features_cache.csv")
    df_merged = df_merged.dropna()
    feature_cols = [
        "sqrt_area_px", "area_norm", "perimeter_px", "circularity",
        "aspect_ratio", "extent", "solidity",
        "depth_in_mean", "depth_in_std", "depth_bg_mean",
        "depth_contrast", "depth_ratio",
        "proxy_linear_in", "proxy_linear_bg", "proxy_area_in", "proxy_relief",
        "area_variability_std", "max_sqrt_area_px", "max_proxy_linear",
        "paris_class_num", "is_pedunculated", "patient_age",
        "patient_gender_num", "endoscope_hq290I"
    ]

    X = df_merged[feature_cols].values
    y = df_merged["Polyp_Size_mm"].values
    n = len(y)
    print(f"=== AUDITING COMPLETE {n} PATIENTS FROM CACHE ===\n")

    # 1. FIXED DOMAIN FEATURES (ZERO SELECTION -> ZERO LEAKAGE)
    domain_cols = ["sqrt_area_px", "proxy_linear_bg", "endoscope_hq290I"]
    fixed_indices = [feature_cols.index(c) for c in domain_cols]

    preds_fixed = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx][:, fixed_indices])
        X_te = scaler.transform(X[[i]][:, fixed_indices])
        rf = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42)
        rf.fit(X_tr, y[train_idx])
        preds_fixed[i] = rf.predict(X_te)[0]

    mae_f = mean_absolute_error(y, preds_fixed)
    rmse_f = np.sqrt(mean_squared_error(y, preds_fixed))
    r2_f = r2_score(y, preds_fixed)
    mape_f = np.mean(np.abs((y - preds_fixed) / y)) * 100.0
    corr_f = np.corrcoef(y, preds_fixed)[0, 1]

    print("--- METHOD 1: FIXED DOMAIN FEATURES LOOCV (ZERO FEATURE SELECTION) ---")
    print(f"Features: {', '.join(domain_cols)}")
    print(f"  True LOOCV MAE       : {mae_f:.3f} mm")
    print(f"  True LOOCV RMSE      : {rmse_f:.3f} mm")
    print(f"  True LOOCV R^2 Score : {r2_f:.3f}")
    print(f"  True LOOCV MAPE      : {mape_f:.1f} %")
    print(f"  Pearson Correlation  : {corr_f:.3f}\n")

    # 2. TRUE NESTED LOOCV (INNER FORWARD SELECTION STRICTLY INSIDE FOLD)
    preds_nested = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_tr_full = X[train_idx]
        y_tr_full = y[train_idx]

        sel = []
        avail = list(range(len(feature_cols)))
        for step in range(2):
            best_e = float("inf")
            best_c = None
            for c in avail:
                cand_sub = sel + [c]
                inner_preds = np.zeros(len(train_idx))
                for k in range(len(train_idx)):
                    in_tr = [m for m in range(len(train_idx)) if m != k]
                    scaler = StandardScaler()
                    xt = scaler.fit_transform(X_tr_full[in_tr][:, cand_sub])
                    xv = scaler.transform(X_tr_full[[k]][:, cand_sub])
                    rf = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=42)
                    rf.fit(xt, y_tr_full[in_tr])
                    inner_preds[k] = rf.predict(xv)[0]
                e = mean_absolute_error(y_tr_full, inner_preds)
                if e < best_e:
                    best_e = e
                    best_c = c
            if best_c is not None:
                sel.append(best_c)
                avail.remove(best_c)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_full[:, sel])
        X_te = scaler.transform(X[[i]][:, sel])
        rf = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42)
        rf.fit(X_tr, y_tr_full)
        preds_nested[i] = rf.predict(X_te)[0]

    mae_n = mean_absolute_error(y, preds_nested)
    rmse_n = np.sqrt(mean_squared_error(y, preds_nested))
    r2_n = r2_score(y, preds_nested)
    mape_n = np.mean(np.abs((y - preds_nested) / y)) * 100.0
    corr_n = np.corrcoef(y, preds_nested)[0, 1]

    print("--- METHOD 2: TRUE NESTED LOOCV (ZERO LEAKAGE) ---")
    print(f"  True Nested MAE       : {mae_n:.3f} mm")
    print(f"  True Nested RMSE      : {rmse_n:.3f} mm")
    print(f"  True Nested R^2 Score : {r2_n:.3f}")
    print(f"  True Nested MAPE      : {mape_n:.1f} %")
    print(f"  Pearson Correlation   : {corr_n:.3f}")

if __name__ == "__main__":
    main()
