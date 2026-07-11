"""
Complete 42-Subject Polyp Size Estimation Pipeline via Monocular Endoscopy.
Includes Automatic Fallback for filtered videos + Caching + Strict Zero-Leakage Audit.

Validation Strategy:
  1. Fixed Domain Features LOOCV (Zero Feature Selection -> Zero Leakage)
  2. True Nested LOOCV (Inner feature selection done strictly inside fold -> Zero Leakage)
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.depthpolyp import build_depthpolyp


def load_labels(labels_path: str) -> pd.DataFrame:
    df = pd.read_excel(labels_path)
    df["Polyp_Size_mm"] = df["Polyp_Size"].astype(str).str.replace("mm", "").str.strip().astype(float)
    df["Video_ID"] = df["Video_ID"].astype(int)
    return df


def extract_video_features(
    video_id: int,
    orig_dir: str,
    mask_dir: str,
    model,
    device,
    pw: int = 400,
    ph: int = 340
) -> dict:
    vname = f"Video{video_id:02d}.mp4"
    mask_name = f"Video{video_id:02d}_aggressive.mp4"
    orig_path = os.path.join(orig_dir, vname)
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(orig_path):
        return None

    active_frames = []
    # Check aggressive mask first
    if os.path.exists(mask_path):
        cap_mask = cv2.VideoCapture(mask_path)
        frame_idx = 0
        while True:
            ok = cap_mask.grab()
            if not ok:
                break
            if frame_idx % 10 == 0:
                ok, frame = cap_mask.retrieve()
                if ok:
                    mask_panel = frame[:, pw:pw*2, 0]
                    area = np.sum(mask_panel > 128)
                    if area > 50:
                        active_frames.append((frame_idx, area))
            frame_idx += 1
        cap_mask.release()

    # Fallback to raw DepthPolyp segmentation if aggressive mask had no active frames
    use_fallback = len(active_frames) == 0
    if use_fallback:
        cap_orig = cv2.VideoCapture(orig_path)
        frame_idx = 0
        while True:
            ok = cap_orig.grab()
            if not ok:
                break
            if frame_idx % 10 == 0:
                ok, f = cap_orig.retrieve()
                if ok:
                    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    inp = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
                    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
                    with torch.no_grad():
                        seg, _ = model(t)
                    prob = torch.sigmoid(seg)[0, 0].cpu().numpy()
                    area = np.sum(prob > 0.35)
                    if area > 50:
                        active_frames.append((frame_idx, area))
            frame_idx += 1
        cap_orig.release()

    if not active_frames:
        return None

    active_frames.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in active_frames[:3]]

    cap_orig = cv2.VideoCapture(orig_path)
    cap_mask = cv2.VideoCapture(mask_path) if (os.path.exists(mask_path) and not use_fallback) else None

    f_sqrt_areas = []
    f_areas_norm = []
    f_perimeters = []
    f_circularities = []
    f_aspect_ratios = []
    f_extents = []
    f_solidities = []
    f_depth_in = []
    f_depth_in_std = []
    f_depth_bg = []
    f_depth_contrast = []
    f_depth_ratio = []
    f_proxy_linear_in = []
    f_proxy_linear_bg = []
    f_proxy_area_in = []
    f_proxy_relief = []

    for target_idx in top_indices:
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok1, f_orig = cap_orig.read()
        if not ok1:
            continue

        rgb = cv2.cvtColor(f_orig, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            seg, pred_depth = model(tensor)

        depth_map = pred_depth.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map, (pw, ph))

        if use_fallback:
            prob = torch.sigmoid(seg)[0, 0].cpu().numpy()
            prob_res = cv2.resize(prob, (pw, ph))
            mask_panel = (prob_res > 0.35).astype(np.uint8)
        else:
            cap_mask.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ok2, f_mask = cap_mask.read()
            if not ok2:
                continue
            mask_panel = (f_mask[:, pw:pw*2, 0] > 128).astype(np.uint8)

        area_px = np.sum(mask_panel)
        if area_px < 50:
            continue

        d_in = float(np.mean(depth_map[mask_panel == 1]))
        d_in_std = float(np.std(depth_map[mask_panel == 1]))
        d_bg = float(np.mean(depth_map[mask_panel == 0]))
        d_contrast = d_bg - d_in
        d_ratio = d_in / max(d_bg, 1e-4)

        contours, _ = cv2.findContours(mask_panel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            circ = (4.0 * np.pi * area_px) / max(perim**2, 1e-5)

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / max(bh, 1)
            extent = area_px / max(bw * bh, 1)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area_px / max(hull_area, 1e-5)
        else:
            perim, circ, aspect, extent, solidity = 0.0, 0.5, 1.0, 0.5, 0.8

        sqrt_a = np.sqrt(area_px)
        f_sqrt_areas.append(sqrt_a)
        f_areas_norm.append(area_px / (pw * ph))
        f_perimeters.append(perim)
        f_circularities.append(circ)
        f_aspect_ratios.append(aspect)
        f_extents.append(extent)
        f_solidities.append(solidity)

        f_depth_in.append(d_in)
        f_depth_in_std.append(d_in_std)
        f_depth_bg.append(d_bg)
        f_depth_contrast.append(d_contrast)
        f_depth_ratio.append(d_ratio)

        f_proxy_linear_in.append(sqrt_a * d_in)
        f_proxy_linear_bg.append(sqrt_a * d_bg)
        f_proxy_area_in.append(area_px * (d_in ** 2))
        f_proxy_relief.append(sqrt_a * d_contrast)

    cap_orig.release()
    if cap_mask:
        cap_mask.release()

    if not f_sqrt_areas:
        return None

    return {
        "Video_ID": video_id,
        "sqrt_area_px": float(np.median(f_sqrt_areas)),
        "area_norm": float(np.median(f_areas_norm)),
        "perimeter_px": float(np.median(f_perimeters)),
        "circularity": float(np.median(f_circularities)),
        "aspect_ratio": float(np.median(f_aspect_ratios)),
        "extent": float(np.median(f_extents)),
        "solidity": float(np.median(f_solidities)),
        "depth_in_mean": float(np.median(f_depth_in)),
        "depth_in_std": float(np.median(f_depth_in_std)),
        "depth_bg_mean": float(np.median(f_depth_bg)),
        "depth_contrast": float(np.median(f_depth_contrast)),
        "depth_ratio": float(np.median(f_depth_ratio)),
        "proxy_linear_in": float(np.median(f_proxy_linear_in)),
        "proxy_linear_bg": float(np.median(f_proxy_linear_bg)),
        "proxy_area_in": float(np.median(f_proxy_area_in)),
        "proxy_relief": float(np.median(f_proxy_relief)),
        "area_variability_std": float(np.std(f_sqrt_areas)),
        "max_sqrt_area_px": float(np.max(f_sqrt_areas)),
        "max_proxy_linear": float(np.max(f_proxy_linear_in)),
    }


def main():
    labels_path = r"C:\Users\vilia\Downloads\Polyp_Size_Lables.csv"
    orig_dir = r"C:\Users\vilia\Downloads\Polyp_Size_Dataset\Polyp_Size_Videos"
    mask_dir = r"c:\PolypSizer\inference_aggressive"
    ckpt_path = r"checkpoints\DepthPolyp_Kvasir.pth"
    cache_path = "polyp_features_cache.csv"

    print("=" * 75)
    print("COMPLETE 42-SUBJECT POLYP SIZE ESTIMATION & STATISTICAL AUDIT")
    print("=" * 75)

    df_labels = load_labels(labels_path)
    print(f"Loaded {len(df_labels)} ground truth labels from {labels_path}.")

    if os.path.exists(cache_path):
        print(f"Loading cached features from '{cache_path}'...")
        df_merged = pd.read_csv(cache_path)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_depthpolyp(
            encoder_name="b0", in_channels=3, num_classes=2,
            decoder_channels=256, activation=None,
        )
        model.load_pretrained(ckpt_path)
        model.to(device).eval()

        print("Extracting features across all 42 videos (with fallback for aggressive filters)...")
        records = []
        for vid in range(1, 43):
            feat = extract_video_features(vid, orig_dir, mask_dir, model, device)
            if feat is not None:
                records.append(feat)
                if vid % 10 == 0 or vid == 42:
                    print(f"  Processed {vid}/42 videos...")
            else:
                print(f"  Warning: Video {vid} could not be extracted.")

        df_feats = pd.DataFrame(records)
        df_merged = pd.merge(df_labels, df_feats, on="Video_ID", how="inner")

        paris_map = {"IIa": 1, "Is": 2, "Isp": 3, "Ip": 4}
        df_merged["paris_class_num"] = df_merged["Paris_Classification"].map(paris_map).fillna(2)
        df_merged["is_pedunculated"] = df_merged["Paris_Classification"].isin(["Ip", "Isp"]).astype(int)
        df_merged["patient_age"] = df_merged["Age"].astype(float)
        df_merged["patient_gender_num"] = (df_merged["Gender"] == "Male").astype(int)
        df_merged["endoscope_hq290I"] = (df_merged["Endoscope_Model"].str.contains("HQ290I", na=False)).astype(int)

        df_merged.to_csv(cache_path, index=False)
        print(f"Saved extracted features for {len(df_merged)} subjects to '{cache_path}'.")

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
    print(f"\nAudit dataset complete: {n} / 42 patients.\n")

    # -------------------------------------------------------------------------
    # METHOD 1: FIXED DOMAIN FEATURES LOOCV (Zero Feature Selection -> Zero Leakage)
    # -------------------------------------------------------------------------
    print("-" * 75)
    print("AUDIT METHOD 1: FIXED DOMAIN FEATURES LOOCV (Zero Feature Selection)")
    print("-" * 75)
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

    mae_fixed = mean_absolute_error(y, preds_fixed)
    rmse_fixed = np.sqrt(mean_squared_error(y, preds_fixed))
    r2_fixed = r2_score(y, preds_fixed)
    mape_fixed = np.mean(np.abs((y - preds_fixed) / y)) * 100.0
    corr_fixed = np.corrcoef(y, preds_fixed)[0, 1]

    print(f"Fixed Features         : {', '.join(domain_cols)}")
    print(f"  True LOOCV MAE       : {mae_fixed:.3f} mm")
    print(f"  True LOOCV RMSE      : {rmse_fixed:.3f} mm")
    print(f"  True LOOCV R^2 Score : {r2_fixed:.3f}")
    print(f"  True LOOCV MAPE      : {mape_fixed:.1f} %")
    print(f"  Pearson Correlation  : {corr_fixed:.3f}")

    # -------------------------------------------------------------------------
    # METHOD 2: TRUE NESTED LOOCV (Inner Selection inside fold -> Zero Leakage)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("AUDIT METHOD 2: TRUE NESTED LOOCV (Feature selection strictly inside outer fold)")
    print("-" * 75)
    preds_nested = np.zeros(n)
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_tr_full = X[train_idx]
        y_tr_full = y[train_idx]

        # Inner forward selection on 41 training samples only
        sel = []
        avail = list(range(len(feature_cols)))
        for step in range(2): # select top 2 features strictly inside fold
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

    mae_nested = mean_absolute_error(y, preds_nested)
    rmse_nested = np.sqrt(mean_squared_error(y, preds_nested))
    r2_nested = r2_score(y, preds_nested)
    mape_nested = np.mean(np.abs((y - preds_nested) / y)) * 100.0
    corr_nested = np.corrcoef(y, preds_nested)[0, 1]

    print(f"  True Nested MAE      : {mae_nested:.3f} mm")
    print(f"  True Nested RMSE     : {rmse_nested:.3f} mm")
    print(f"  True Nested R^2 Score: {r2_nested:.3f}")
    print(f"  True Nested MAPE     : {mape_nested:.1f} %")
    print(f"  Pearson Correlation  : {corr_nested:.3f}")

    # Save complete honest 42-patient LOOCV predictions
    df_out = df_merged[["Video_ID", "Polyp_Size_mm"]].copy()
    df_out["Predicted_Size_Fixed_mm"] = np.round(preds_fixed, 2)
    df_out["AbsError_Fixed_mm"] = np.round(np.abs(df_out["Polyp_Size_mm"] - df_out["Predicted_Size_Fixed_mm"]), 2)
    df_out["Predicted_Size_Nested_mm"] = np.round(preds_nested, 2)
    df_out["AbsError_Nested_mm"] = np.round(np.abs(df_out["Polyp_Size_mm"] - df_out["Predicted_Size_Nested_mm"]), 2)
    df_out.to_csv("polyp_size_loocv_predictions_complete42.csv", index=False)
    print("\nSaved complete 42-patient honest audit predictions to 'polyp_size_loocv_predictions_complete42.csv'.")


if __name__ == "__main__":
    main()
