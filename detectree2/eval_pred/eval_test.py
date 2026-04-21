"""
Standalone COCO-style AP evaluation for tree crown detection.
Does not rely on pycocotools' coordinate handling - computes IoU directly from geometries.
"""

import argparse
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from collections import defaultdict


def compute_iou_matrix(gt_gdf, pred_gdf):
    """
    Compute IoU matrix between all GT and prediction pairs.
    Returns: (n_gt, n_pred) matrix of IoU values
    """
    n_gt = len(gt_gdf)
    n_pred = len(pred_gdf)

    iou_matrix = np.zeros((n_gt, n_pred))

    # Build spatial index for predictions
    pred_sindex = pred_gdf.sindex

    for gt_idx, gt_row in enumerate(gt_gdf.itertuples()):
        gt_geom = gt_row.geometry
        if gt_geom is None or gt_geom.is_empty:
            continue

        # Find candidate predictions using spatial index
        candidates = list(pred_sindex.intersection(gt_geom.bounds))

        for pred_idx in candidates:
            pred_geom = pred_gdf.iloc[pred_idx].geometry
            if pred_geom is None or pred_geom.is_empty:
                continue

            if gt_geom.intersects(pred_geom):
                try:
                    intersection = gt_geom.intersection(pred_geom).area
                    union = gt_geom.union(pred_geom).area
                    iou = intersection / union if union > 0 else 0
                    iou_matrix[gt_idx, pred_idx] = iou
                except Exception:
                    pass

    return iou_matrix


def compute_ap_at_iou(iou_matrix, scores, iou_thresh):
    """
    Compute AP at a specific IoU threshold.

    Args:
        iou_matrix: (n_gt, n_pred) IoU values
        scores: (n_pred,) confidence scores
        iou_thresh: IoU threshold for matching

    Returns:
        AP value
    """
    n_gt, n_pred = iou_matrix.shape

    if n_gt == 0 or n_pred == 0:
        return 0.0

    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-scores)

    # Track which GT have been matched
    gt_matched = np.zeros(n_gt, dtype=bool)

    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    for i, pred_idx in enumerate(sorted_indices):
        # Find best matching GT for this prediction
        ious = iou_matrix[:, pred_idx]

        # Only consider unmatched GT with IoU >= threshold
        valid_mask = (~gt_matched) & (ious >= iou_thresh)

        if valid_mask.any():
            # Match with highest IoU GT
            best_gt = np.argmax(ious * valid_mask)
            if ious[best_gt] >= iou_thresh:
                tp[i] = 1
                gt_matched[best_gt] = True
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Compute AP using 101-point interpolation (COCO style)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recall >= t
        if mask.any():
            ap += np.max(precision[mask]) / 101

    return ap


def compute_f1_at_iou(iou_matrix, scores, iou_thresh, conf_thresh=0.5):
    """
    Compute precision, recall, F1 at specific IoU and confidence thresholds.
    """
    n_gt, n_pred = iou_matrix.shape

    if n_gt == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'tp': 0, 'fp': 0, 'fn': n_gt}

    # Filter by confidence
    conf_mask = scores >= conf_thresh

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = 0
    fp = 0

    # Sort by confidence
    sorted_indices = np.argsort(-scores)

    for pred_idx in sorted_indices:
        if not conf_mask[pred_idx]:
            continue

        ious = iou_matrix[:, pred_idx]
        valid_mask = (~gt_matched) & (ious >= iou_thresh)

        if valid_mask.any():
            best_gt = np.argmax(ious * valid_mask)
            if ious[best_gt] >= iou_thresh:
                tp += 1
                gt_matched[best_gt] = True
            else:
                fp += 1
        else:
            fp += 1

    fn = n_gt - gt_matched.sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def run_evaluation(gt_gpkg_path, pred_gpkg_path, output_path=None, clip_to_gt=True):
    """
    Run full COCO-style evaluation.
    """
    print("="*60)
    print("CROWN DETECTION EVALUATION")
    print("="*60)

    # Load data
    print("\nLoading ground truth...")
    gt_gdf = gpd.read_file(gt_gpkg_path)
    print(f"  {len(gt_gdf)} ground truth crowns")

    print("\nLoading predictions...")
    pred_gdf = gpd.read_file(pred_gpkg_path)
    print(f"  {len(pred_gdf)} predicted crowns")

    # Find confidence score column
    score_col = None
    for col in ['Confidence_score', 'score', 'confidence', 'Score']:
        if col in pred_gdf.columns:
            score_col = col
            break

    if score_col:
        print(f"  Using '{score_col}' as confidence score")
        scores = pred_gdf[score_col].values
    else:
        print("  WARNING: No confidence score found, using 1.0 for all")
        scores = np.ones(len(pred_gdf))

    # Clip predictions to GT extent
    if clip_to_gt:
        print("\nClipping predictions to GT extent...")
        gt_bounds = gt_gdf.total_bounds
        gt_bbox = box(*gt_bounds)
        mask = pred_gdf.geometry.intersects(gt_bbox)
        pred_gdf = pred_gdf[mask].copy()
        scores = scores[mask]
        print(f"  {len(pred_gdf)} predictions within GT extent")

    # Reset indices
    gt_gdf = gt_gdf.reset_index(drop=True)
    pred_gdf = pred_gdf.reset_index(drop=True)

    # Compute IoU matrix
    print("\nComputing IoU matrix...")
    iou_matrix = compute_iou_matrix(gt_gdf, pred_gdf)
    print(f"  Shape: {iou_matrix.shape}")
    print(f"  Non-zero IoUs: {(iou_matrix > 0).sum()}")
    print(f"  Max IoU: {iou_matrix.max():.3f}")

    # Compute AP at multiple IoU thresholds
    print("\nComputing AP at various IoU thresholds...")
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []

    for iou_thresh in iou_thresholds:
        ap = compute_ap_at_iou(iou_matrix, scores, iou_thresh)
        aps.append(ap)
        print(f"  AP @ IoU={iou_thresh:.2f}: {ap:.3f}")

    # Summary metrics
    ap_50_95 = np.mean(aps)
    ap_50 = aps[0]  # IoU = 0.50
    ap_75 = aps[5]  # IoU = 0.75

    # Compute F1 metrics
    print("\nComputing F1 metrics...")
    f1_results = {}
    for conf in [0.3, 0.5, 0.7]:
        f1_50 = compute_f1_at_iou(iou_matrix, scores, iou_thresh=0.5, conf_thresh=conf)
        f1_results[f'conf_{conf}'] = f1_50
        print(f"  Conf>={conf}, IoU>=0.5: P={f1_50['precision']:.3f}, R={f1_50['recall']:.3f}, F1={f1_50['f1']:.3f}")

    # Area-based analysis
    print("\nArea-based analysis...")
    gt_areas = gt_gdf.geometry.area.values
    pred_areas = pred_gdf.geometry.area.values

    # Define area thresholds (in m²)
    area_small = 10  # < 10 m²
    area_large = 50  # > 50 m²

    for area_name, area_min, area_max in [('small', 0, area_small),
                                           ('medium', area_small, area_large),
                                           ('large', area_large, 1e10)]:
        gt_mask = (gt_areas >= area_min) & (gt_areas < area_max)
        n_gt_area = gt_mask.sum()

        if n_gt_area > 0:
            # Filter IoU matrix for this area range
            iou_sub = iou_matrix[gt_mask, :]
            ap_area = compute_ap_at_iou(iou_sub, scores, 0.5)
            print(f"  {area_name} ({area_min}-{area_max} m²): {n_gt_area} GT, AP50={ap_area:.3f}")
        else:
            print(f"  {area_name} ({area_min}-{area_max} m²): 0 GT")

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nDataset:")
    print(f"  Ground truth crowns: {len(gt_gdf)}")
    print(f"  Predicted crowns:    {len(pred_gdf)}")
    print(f"  Pred/GT ratio:       {len(pred_gdf)/len(gt_gdf):.2f}")

    print(f"\nCOCO-style metrics:")
    print(f"  AP (IoU=0.50:0.95): {ap_50_95:.3f}")
    print(f"  AP50:               {ap_50:.3f}")
    print(f"  AP75:               {ap_75:.3f}")

    print(f"\nF1 @ IoU=0.5, Conf>=0.5:")
    f1_default = f1_results['conf_0.5']
    print(f"  Precision: {f1_default['precision']:.3f}")
    print(f"  Recall:    {f1_default['recall']:.3f}")
    print(f"  F1:        {f1_default['f1']:.3f}")
    print(f"  TP={f1_default['tp']}, FP={f1_default['fp']}, FN={f1_default['fn']}")

    # Compile results (convert numpy types to native Python for JSON)
    def to_native(x):
        if isinstance(x, (np.integer, np.int64, np.int32)):
            return int(x)
        elif isinstance(x, (np.floating, np.float64, np.float32)):
            return float(x)
        return x

    results = {
        'coco_metrics': {
            'AP': float(ap_50_95),
            'AP50': float(ap_50),
            'AP75': float(ap_75),
        },
        'ap_per_iou': {f'AP{int(t*100)}': float(ap) for t, ap in zip(iou_thresholds, aps)},
        'f1_metrics': {k: {kk: to_native(vv) for kk, vv in v.items()}
                      for k, v in f1_results.items()},
        'metadata': {
            'n_gt': int(len(gt_gdf)),
            'n_pred': int(len(pred_gdf)),
            'gt_file': gt_gpkg_path,
            'pred_file': pred_gpkg_path,
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Crown detection evaluation')
    parser.add_argument('--gt', required=True, help='Ground truth gpkg')
    parser.add_argument('--pred', required=True, help='Prediction gpkg')
    parser.add_argument('--output', default='eval_results.json', help='Output JSON')
    parser.add_argument('--no-clip', action='store_true', help='Do not clip to GT extent')

    args = parser.parse_args()

    run_evaluation(
        gt_gpkg_path=args.gt,
        pred_gpkg_path=args.pred,
        output_path=args.output,
        clip_to_gt=not args.no_clip
    )


if __name__ == '__main__':
    main()