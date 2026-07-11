"""
Polyp Size Binary Classification Pipeline (< 5 mm vs >= 5 mm).
Strict Leave-One-Out Cross-Validation (LOOCV) on all 42 subjects.

Compares:
  1. Direct Binary Classifiers: Random Forest Classifier, Logistic Regression
  2. Thresholded Regression Predictions (from our honest Random Forest Regressor)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def main():
    print("=" * 75)
    print("POLYP SIZE BINARY CLASSIFICATION (< 5 mm vs >= 5 mm) - LOOCV ON 42 SUBJECTS")
    print("=" * 75)

    df = pd.read_csv("polyp_features_cache.csv")
    df = df.fillna(df.median(numeric_only=True))

    feature_cols = ["sqrt_area_px", "proxy_linear_bg", "endoscope_hq290I"]
    X = df[feature_cols].values
    y_continuous = df["Polyp_Size_mm"].values
    y_binary = (y_continuous >= 5.0).astype(int)  # 1 = >= 5mm, 0 = < 5mm

    n = len(y_binary)
    n_pos = np.sum(y_binary == 1)
    n_neg = np.sum(y_binary == 0)
    print(f"Dataset composition: {n} total subjects")
    print(f"  Class 0 (< 5 mm)  : {n_neg} subjects ({n_neg/n*100:.1f}%)")
    print(f"  Class 1 (>= 5 mm) : {n_pos} subjects ({n_pos/n*100:.1f}%)\n")

    # 1. DIRECT RANDOM FOREST CLASSIFIER (LOOCV)
    preds_rfc = np.zeros(n, dtype=int)
    probs_rfc = np.zeros(n)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[[i]])

        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_tr, y_binary[train_idx])
        preds_rfc[i] = clf.predict(X_te)[0]
        probs_rfc[i] = clf.predict_proba(X_te)[0, 1]

    acc_rfc = accuracy_score(y_binary, preds_rfc) * 100.0
    cm_rfc = confusion_matrix(y_binary, preds_rfc)
    tn, fp, fn, tp = cm_rfc.ravel()
    sens_rfc = (tp / max(tp + fn, 1)) * 100.0
    spec_rfc = (tn / max(tn + fp, 1)) * 100.0
    auc_rfc = roc_auc_score(y_binary, probs_rfc)

    print("-" * 75)
    print("1. DIRECT RANDOM FOREST CLASSIFIER (LOOCV):")
    print("-" * 75)
    print(f"  Accuracy    : {acc_rfc:.1f} %  ({tp+tn}/{n} correctly classified)")
    print(f"  Sensitivity : {sens_rfc:.1f} %  (Detecting polyp >= 5 mm)")
    print(f"  Specificity : {spec_rfc:.1f} %  (Detecting polyp < 5 mm)")
    print(f"  ROC-AUC     : {auc_rfc:.3f}")
    print(f"  Confusion Matrix :\n    TN (<5 pred <5): {tn}  |  FP (<5 pred >=5): {fp}\n    FN (>=5 pred <5): {fn}  |  TP (>=5 pred >=5): {tp}\n")

    # 2. LOGISTIC REGRESSION CLASSIFIER (LOOCV)
    preds_lr = np.zeros(n, dtype=int)
    probs_lr = np.zeros(n)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[[i]])

        clf = LogisticRegression(C=1.0, random_state=42)
        clf.fit(X_tr, y_binary[train_idx])
        preds_lr[i] = clf.predict(X_te)[0]
        probs_lr[i] = clf.predict_proba(X_te)[0, 1]

    acc_lr = accuracy_score(y_binary, preds_lr) * 100.0
    cm_lr = confusion_matrix(y_binary, preds_lr)
    tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
    sens_lr = (tp_lr / max(tp_lr + fn_lr, 1)) * 100.0
    spec_lr = (tn_lr / max(tn_lr + fp_lr, 1)) * 100.0
    auc_lr = roc_auc_score(y_binary, probs_lr)

    print("-" * 75)
    print("2. LOGISTIC REGRESSION CLASSIFIER (LOOCV):")
    print("-" * 75)
    print(f"  Accuracy    : {acc_lr:.1f} %  ({tp_lr+tn_lr}/{n} correctly classified)")
    print(f"  Sensitivity : {sens_lr:.1f} %")
    print(f"  Specificity : {spec_lr:.1f} %")
    print(f"  ROC-AUC     : {auc_lr:.3f}")
    print(f"  Confusion Matrix :\n    TN (<5 pred <5): {tn_lr}  |  FP (<5 pred >=5): {fp_lr}\n    FN (>=5 pred <5): {fn_lr}  |  TP (>=5 pred >=5): {tp_lr}\n")

    # 3. THRESHOLDED CONTINUOUS RANDOM FOREST REGRESSION PREDICTIONS (LOOCV)
    preds_reg = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[[i]])

        reg = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42)
        reg.fit(X_tr, y_continuous[train_idx])
        preds_reg[i] = reg.predict(X_te)[0]

    preds_reg_bin = (preds_reg >= 5.0).astype(int)
    acc_reg = accuracy_score(y_binary, preds_reg_bin) * 100.0
    cm_reg = confusion_matrix(y_binary, preds_reg_bin)
    tn_r, fp_r, fn_r, tp_r = cm_reg.ravel()
    sens_reg = (tp_r / max(tp_r + fn_r, 1)) * 100.0
    spec_reg = (tn_r / max(tn_r + fp_r, 1)) * 100.0

    print("-" * 75)
    print("3. THRESHOLDED CONTINUOUS REGRESSION (Pred_mm >= 5.0 mm):")
    print("-" * 75)
    print(f"  Accuracy    : {acc_reg:.1f} %  ({tp_r+tn_r}/{n} correctly classified)")
    print(f"  Sensitivity : {sens_reg:.1f} %")
    print(f"  Specificity : {spec_reg:.1f} %")
    print(f"  Confusion Matrix :\n    TN (<5 pred <5): {tn_r}  |  FP (<5 pred >=5): {fp_r}\n    FN (>=5 pred <5): {fn_r}  |  TP (>=5 pred >=5): {tp_r}\n")

    # Save detailed classification results
    df_out = df[["Video_ID", "Polyp_Size_mm"]].copy()
    df_out["True_Class_ge5mm"] = y_binary
    df_out["RF_Classifier_Pred"] = preds_rfc
    df_out["RF_Classifier_Prob"] = np.round(probs_rfc, 3)
    df_out["Regression_Pred_mm"] = np.round(preds_reg, 2)
    df_out["Regression_Class_Pred"] = preds_reg_bin
    df_out.to_csv("polyp_size_loocv_classification_results.csv", index=False)
    print("Detailed classification results saved to 'polyp_size_loocv_classification_results.csv'.")


if __name__ == "__main__":
    main()
