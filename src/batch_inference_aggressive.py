"""
Aggressive Clinical Batch Inference Pipeline for DepthPolyp.

Runs ONLY on the FIRST 5 VIDEOS.

Aggressive Filtering Rules:
  1. Strong Spatial Center Prior: center weight 1.05, edges drop sharply to 0.25.
  2. Conservative Probability EMA: alpha = 0.25, higher binarization threshold = 0.52.
  3. Strict Temporal Majority Voting: requires at least 4 out of 5 frames (80%).
  4. Aggressive Connected Components & IoU Spatial Lock:
     - Minimum component area = 0.8% of image (removes small/medium noise blobs).
     - Strict spatial continuity lock: peripheral components outside radius 0.50
       are ignored unless locked by ongoing IoU tracking.
"""
import os
import cv2
import numpy as np
import torch
from collections import deque
from model.depthpolyp import build_depthpolyp


class AggressiveClinicalPipeline:
    """
    Hyper-aggressive multi-stage stabilization pipeline for polyp segmentation.
    """

    def __init__(
        self,
        frame_size: tuple[int, int] = (400, 340),
        ema_alpha: float = 0.25,
        seg_threshold: float = 0.52,
        window_size: int = 5,
        majority_votes: int = 4,
        min_area_ratio: float = 0.008,
    ):
        self.w, self.h = frame_size
        self.ema_alpha = ema_alpha
        self.seg_threshold = seg_threshold
        self.window_size = window_size
        self.majority_votes = majority_votes
        self.min_area = int(self.w * self.h * min_area_ratio)

        # Build strong radial center prior map (drops sharply from 1.05 to 0.25 at edges)
        self.center_weight_map = self._build_strong_center_prior(self.h, self.w)

        self.ema_prob = None
        self.mask_buffer = deque(maxlen=window_size)
        self.prev_tracked_mask = None

    @staticmethod
    def _build_strong_center_prior(h: int, w: int) -> np.ndarray:
        y = np.linspace(-1.0, 1.0, h)
        x = np.linspace(-1.0, 1.0, w)
        xx, yy = np.meshgrid(x, y)
        r2 = xx**2 + yy**2
        # Sharp Gaussian decay from center
        gauss = np.exp(-1.5 * r2)
        weight_map = 0.25 + 0.80 * gauss
        return weight_map.astype(np.float32)

    def reset(self):
        self.ema_prob = None
        self.mask_buffer.clear()
        self.prev_tracked_mask = None

    def process(self, raw_prob: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = raw_prob.shape

        # 1) Modulate with aggressive center prior
        weighted_prob = np.clip(raw_prob * self.center_weight_map, 0.0, 1.0)

        # 2) Probability EMA
        if self.ema_prob is None:
            self.ema_prob = weighted_prob.copy()
        else:
            self.ema_prob = (
                self.ema_alpha * weighted_prob
                + (1.0 - self.ema_alpha) * self.ema_prob
            )

        bin_mask = (self.ema_prob >= self.seg_threshold).astype(np.uint8)

        # 3) Strict Temporal Majority Voting (4 out of 5 frames)
        self.mask_buffer.append(bin_mask)
        if len(self.mask_buffer) < self.window_size:
            req_votes = max(1, int(np.ceil(self.majority_votes * len(self.mask_buffer) / self.window_size)))
        else:
            req_votes = self.majority_votes

        vote_sum = np.sum(self.mask_buffer, axis=0)
        voted_mask = (vote_sum >= req_votes).astype(np.uint8) * 255

        # 4) Connected Components + Aggressive IoU Spatial Lock
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            voted_mask, connectivity=8
        )

        final_mask = np.zeros((h, w), dtype=np.uint8)
        tracked = False

        if num_labels > 1:
            center_x, center_y = w / 2.0, h / 2.0
            max_dist = np.hypot(center_x, center_y)

            candidates = []
            for label_idx in range(1, num_labels):
                area = stats[label_idx, cv2.CC_STAT_AREA]
                if area < self.min_area:
                    continue

                comp_mask = (labels == label_idx).astype(np.uint8) * 255
                cx, cy = centroids[label_idx]
                dist_center = np.hypot(cx - center_x, cy - center_y) / max_dist

                iou = 0.0
                if self.prev_tracked_mask is not None:
                    inter = np.logical_and(comp_mask > 0, self.prev_tracked_mask > 0).sum()
                    uni = np.logical_or(comp_mask > 0, self.prev_tracked_mask > 0).sum()
                    iou = inter / max(uni, 1)

                candidates.append({
                    "label": label_idx,
                    "mask": comp_mask,
                    "area": area,
                    "dist_center": dist_center,
                    "iou": iou,
                    "centroid": (cx, cy),
                })

            if candidates:
                if self.prev_tracked_mask is not None:
                    # Spatial lock: only accept candidates overlapping or very close to previous track
                    overlapping = [c for c in candidates if c["iou"] > 0.02]
                    if overlapping:
                        for c in overlapping:
                            final_mask = cv2.bitwise_or(final_mask, c["mask"])
                        tracked = True
                else:
                    # Initial acquisition: only accept central components (dist_center < 0.55)
                    central_candidates = [c for c in candidates if c["dist_center"] < 0.55]
                    if central_candidates:
                        best_c = max(central_candidates, key=lambda c: c["area"])
                        final_mask = best_c["mask"]
                        tracked = True

        self.prev_tracked_mask = final_mask.copy() if np.sum(final_mask) > 0 else None

        return final_mask, {"tracked": tracked}


def colorize_depth(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob, 0.0, 1.0)
    stops = np.array(
        [[84, 5, 38], [132, 33, 86], [140, 48, 141],
         [119, 71, 203], [48, 135, 245], [37, 231, 252]],
        dtype=np.float32,
    )
    s = prob * (len(stops) - 1)
    lo = np.floor(s).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    a = (s - lo)[..., None]
    return (stops[lo] * (1.0 - a) + stops[hi] * a).astype(np.uint8)


def main():
    input_dir = r"C:\Users\vilia\Downloads\Polyp_Size_Dataset\Polyp_Size_Videos"
    ckpt_path = r"checkpoints\DepthPolyp_Kvasir.pth"
    output_dir = r"c:\PolypSizer\inference_aggressive"

    os.makedirs(output_dir, exist_ok=True)
    pw, ph = 400, 340

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_depthpolyp(
        encoder_name="b0", in_channels=3, num_classes=2,
        decoder_channels=256, activation=None,
    )
    model.load_pretrained(ckpt_path)
    model.to(device).eval()

    all_videos = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".mp4"))
    # ONLY PROCESS FIRST 5 VIDEOS
    videos = all_videos[5:]
    print(f"Processing ONLY FIRST {len(videos)} videos out of {len(all_videos)}.\n")

    pipeline = AggressiveClinicalPipeline(
        frame_size=(pw, ph),
        ema_alpha=0.25,
        seg_threshold=0.52,
        window_size=5,
        majority_votes=4,
        min_area_ratio=0.008,
    )

    for idx, vname in enumerate(videos, 1):
        stem = os.path.splitext(vname)[0]
        out_name = f"{stem}_aggressive.mp4"
        out_path = os.path.join(output_dir, out_name)

        cap = cv2.VideoCapture(os.path.join(input_dir, vname))
        if not cap.isOpened():
            print(f"[{idx}/{len(videos)}] Cannot open {vname}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{idx}/{len(videos)}] Processing {vname} ({total} frames)...")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (pw * 3, ph))
        pipeline.reset()

        frame_idx = 0
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                panel_orig = cv2.resize(frame, (pw, ph))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inp = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
                tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
                pred_seg, pred_depth = model(tensor)

                seg_raw = pred_seg.squeeze().cpu().numpy()
                seg_resized = cv2.resize(seg_raw, (pw, ph))

                clean_mask, diag = pipeline.process(seg_resized)

                depth_raw = pred_depth.squeeze().cpu().numpy()
                depth_panel = colorize_depth(cv2.resize(depth_raw, (pw, ph)))

                overlay = panel_orig.copy()
                if clean_mask.sum() > 0:
                    overlay[clean_mask > 0] = (
                        overlay[clean_mask > 0] * 0.6
                        + np.array([37, 231, 252], dtype=np.float64) * 0.4
                    ).astype(np.uint8)

                mask_panel = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

                status_text = "LOCKED (AGGRESSIVE)" if diag["tracked"] else "SEARCHING"
                color = (0, 255, 0) if diag["tracked"] else (0, 160, 255)
                cv2.putText(overlay, f"AGGRESSIVE PIPELINE | {status_text}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

                writer.write(cv2.hconcat([overlay, mask_panel, depth_panel]))

                frame_idx += 1
                if frame_idx % 100 == 0 or frame_idx == total:
                    print(f"  frame {frame_idx}/{total}")

        cap.release()
        writer.release()
        print(f"  -> Saved {out_name}")

    print("\nFirst 5 videos aggressive processing completed successfully!")


if __name__ == "__main__":
    import torch._dynamo
    main()
