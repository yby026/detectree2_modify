"""
Detectree2-style evaluation for merged gpkg files.

Uses the same "bidirectional best-match" logic as detectree2's evaluate.py,
but works directly with merged gpkg files instead of per-tile geojsons.

Key difference from standard COCO evaluation:
- A match is only counted as TP if pred and GT are EACH OTHER'S best IoU match
- This is stricter and handles over-segmentation better
"""

import argparse
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely import make_valid
from shapely.errors import GEOSException


def find_intersections_gpkg(gt_gdf, pred_gdf):
    """
    Find best IoU match for each GT and each prediction.
    Returns two arrays:
      - gt_best_iou[i]: best IoU for GT crown i, and index of matching pred
      - pred_best_iou[j]: best IoU for pred crown j, and index of matching GT

    This mirrors detectree2's find_intersections() logic.
    """
    n_gt = len(gt_gdf)
    n_pred = len(pred_gdf)

    # Track best IoU and matching index for each feature
    gt_best_iou = np.zeros(n_gt)
    gt_best_match = np.full(n_gt, -1, dtype=int)

    pred_best_iou = np.zeros(n_pred)
    pred_best_match = np.full(n_pred, -1, dtype=int)

    # Build spatial index for GT
    gt_sindex = gt_gdf.sindex

    # For each prediction, find intersecting GTs
    for pred_idx in range(n_pred):
        pred_geom = pred_gdf.iloc[pred_idx].geometry

        if pred_geom is None or pred_geom.is_empty:
            continue

        if not pred_geom.is_valid:
            pred_geom = make_valid(pred_geom)

        # Find candidate GTs using spatial index
        candidates = list(gt_sindex.intersection(pred_geom.bounds))

        for gt_idx in candidates:
            gt_geom = gt_gdf.iloc[gt_idx].geometry

            if gt_geom is None or gt_geom.is_empty:
                continue

            if not gt_geom.is_valid:
                gt_geom = make_valid(gt_geom)

            if not gt_geom.intersects(pred_geom):
                continue

            try:
                intersection = pred_geom.intersection(gt_geom).area
                union_area = pred_geom.union(gt_geom).area
                iou = intersection / union_area if union_area > 0 else 0
            except (GEOSException, Exception) as e:
                continue

            # Update best IoU for GT
            if iou > gt_best_iou[gt_idx]:
                gt_best_iou[gt_idx] = iou
                gt_best_match[gt_idx] = pred_idx

            # Update best IoU for pred
            if iou > pred_best_iou[pred_idx]:
                pred_best_iou[pred_idx] = iou
                pred_best_match[pred_idx] = gt_idx

    return gt_best_iou, gt_best_match, pred_best_iou, pred_best_match


def compute_tp_fp_fn(gt_best_iou, gt_best_match, pred_best_iou, pred_best_match,
                     iou_threshold=0.5, conf_scores=None, conf_threshold=0.0):
    """
    Compute TP, FP, FN using detectree2's bidirectional matching logic.

    A prediction is a TP only if:
    1. It has a matching GT (pred_best_match != -1)
    2. The IoU >= threshold
    3. The GT's best match is THIS prediction (bidirectional)
    4. Confidence >= threshold (if applicable)
    """
    n_gt = len(gt_best_iou)
    n_pred = len(pred_best_iou)

    # Track which GTs have been matched as TP
    gt_matched = np.zeros(n_gt, dtype=bool)

    tps = 0
    fps = 0

    for pred_idx in range(n_pred):
        # Check confidence threshold
        if conf_scores is not None and conf_scores[pred_idx] < conf_threshold:
            continue  # Skip low-confidence predictions (don't count as FP either)

        gt_idx = pred_best_match[pred_idx]

        # No intersection with any GT
        if gt_idx == -1:
            fps += 1
            continue

        # Check bidirectional match AND IoU threshold
        if (gt_best_match[gt_idx] == pred_idx and
            pred_best_iou[pred_idx] >= iou_threshold):
            # This is a true positive
            tps += 1
            gt_matched[gt_idx] = True
        else:
            # Either not bidirectional best match, or IoU too low
            fps += 1

    # FN = GT crowns that weren't matched
    fns = n_gt - gt_matched.sum()

    return tps, fps, fns, gt_matched


def run_dt2_eval(gt_gpkg_path, pred_gpkg_path, output_path=None,
                               iou_threshold=0.5, conf_threshold=0.0,
                               area_threshold=0.0, clip_to_gt=True):
    """
    Run detectree2-style evaluation on merged gpkg files.
    """
    print("="*60)
    print("DETECTREE2-STYLE EVALUATION")
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
        conf_scores = pred_gdf[score_col].values
    else:
        print("  No confidence score found")
        conf_scores = None

    # Clip predictions to GT extent
    if clip_to_gt:
        print("\nClipping predictions to GT extent...")
        gt_bounds = gt_gdf.total_bounds
        gt_bbox = box(*gt_bounds)
        mask = pred_gdf.geometry.intersects(gt_bbox)
        pred_gdf = pred_gdf[mask].copy()
        if conf_scores is not None:
            conf_scores = conf_scores[mask]
        print(f"  {len(pred_gdf)} predictions within GT extent")

    # Filter by area threshold
    if area_threshold > 0:
        print(f"\nFiltering by area threshold ({area_threshold} m²)...")
        gt_mask = gt_gdf.geometry.area >= area_threshold
        pred_mask = pred_gdf.geometry.area >= area_threshold
        gt_gdf = gt_gdf[gt_mask].copy()
        pred_gdf = pred_gdf[pred_mask].copy()
        if conf_scores is not None:
            conf_scores = conf_scores[pred_mask]
        print(f"  {len(gt_gdf)} GT, {len(pred_gdf)} pred after filtering")

    # Reset indices
    gt_gdf = gt_gdf.reset_index(drop=True)
    pred_gdf = pred_gdf.reset_index(drop=True)

    # Find intersections (detectree2 style)
    print("\nFinding intersections (bidirectional matching)...")
    gt_best_iou, gt_best_match, pred_best_iou, pred_best_match = \
        find_intersections_gpkg(gt_gdf, pred_gdf)

    print(f"  GT crowns with any intersection: {(gt_best_iou > 0).sum()}")
    print(f"  Pred crowns with any intersection: {(pred_best_iou > 0).sum()}")
    print(f"  Max GT IoU: {gt_best_iou.max():.3f}")
    print(f"  Max Pred IoU: {pred_best_iou.max():.3f}")

    # Compute metrics at different IoU thresholds
    print("\n" + "-"*60)
    print("Results at different IoU thresholds:")
    print("-"*60)

    results_by_iou = {}
    for iou_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        tps, fps, fns, _ = compute_tp_fp_fn(
            gt_best_iou, gt_best_match, pred_best_iou, pred_best_match,
            iou_threshold=iou_thresh, conf_scores=conf_scores, conf_threshold=conf_threshold
        )

        precision = tps / (tps + fps) if (tps + fps) > 0 else 0
        recall = tps / (tps + fns) if (tps + fns) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results_by_iou[f'iou_{iou_thresh}'] = {
            'tp': tps, 'fp': fps, 'fn': fns,
            'precision': precision, 'recall': recall, 'f1': f1
        }

        print(f"  IoU >= {iou_thresh}: TP={tps:4d}, FP={fps:4d}, FN={fns:4d} | "
              f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    # Compute metrics at different confidence thresholds (IoU=0.5)
    print("\n" + "-"*60)
    print("Results at different confidence thresholds (IoU=0.5):")
    print("-"*60)

    results_by_conf = {}
    if conf_scores is not None:
        for conf_thresh in [0.0, 0.3, 0.5, 0.7, 0.9]:
            tps, fps, fns, _ = compute_tp_fp_fn(
                gt_best_iou, gt_best_match, pred_best_iou, pred_best_match,
                iou_threshold=0.5, conf_scores=conf_scores, conf_threshold=conf_thresh
            )

            precision = tps / (tps + fps) if (tps + fps) > 0 else 0
            recall = tps / (tps + fns) if (tps + fns) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            n_pred_above = (conf_scores >= conf_thresh).sum()

            results_by_conf[f'conf_{conf_thresh}'] = {
                'tp': tps, 'fp': fps, 'fn': fns,
                'precision': precision, 'recall': recall, 'f1': f1,
                'n_pred': n_pred_above
            }

            print(f"  Conf >= {conf_thresh}: TP={tps:4d}, FP={fps:4d}, FN={fns:4d} | "
                  f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (n_pred={n_pred_above})")

    # Main result (IoU=0.5, conf=0.0)
    main_result = results_by_iou['iou_0.5']

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (detectree2 style, IoU >= 0.5)")
    print("="*60)
    print(f"\nDataset:")
    print(f"  Ground truth crowns: {len(gt_gdf)}")
    print(f"  Predicted crowns:    {len(pred_gdf)}")
    print(f"  Pred/GT ratio:       {len(pred_gdf)/len(gt_gdf):.2f}")

    print(f"\nMatching statistics:")
    print(f"  True Positives:  {main_result['tp']}")
    print(f"  False Positives: {main_result['fp']}")
    print(f"  False Negatives: {main_result['fn']}")

    print(f"\nMetrics:")
    print(f"  Precision: {main_result['precision']:.3f}")
    print(f"  Recall:    {main_result['recall']:.3f}")
    print(f"  F1 Score:  {main_result['f1']:.3f}")

    # IoU distribution analysis
    print(f"\nIoU distribution (GT best matches):")
    for thresh in [0.3, 0.5, 0.7]:
        count = (gt_best_iou >= thresh).sum()
        pct = count / len(gt_best_iou) * 100
        print(f"  IoU >= {thresh}: {count} ({pct:.1f}%)")

    # Compile results
    def to_native(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        elif isinstance(x, (np.floating,)):
            return float(x)
        return x

    results = {
        'main_result': {k: to_native(v) for k, v in main_result.items()},
        'by_iou_threshold': {k: {kk: to_native(vv) for kk, vv in v.items()}
                            for k, v in results_by_iou.items()},
        'by_conf_threshold': {k: {kk: to_native(vv) for kk, vv in v.items()}
                             for k, v in results_by_conf.items()},
        'metadata': {
            'n_gt': int(len(gt_gdf)),
            'n_pred': int(len(pred_gdf)),
            'gt_file': gt_gpkg_path,
            'pred_file': pred_gpkg_path,
            'iou_threshold': iou_threshold,
            'conf_threshold': conf_threshold,
            'area_threshold': area_threshold
        }
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Detectree2-style crown evaluation')
    parser.add_argument('--gt', required=True, help='Ground truth gpkg')
    parser.add_argument('--pred', required=True, help='Prediction gpkg')
    parser.add_argument('--output', default='detectree2_eval_results.json')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--conf', type=float, default=0.0, help='Confidence threshold')
    parser.add_argument('--area', type=float, default=0.0, help='Min area threshold (m²)')
    parser.add_argument('--no-clip', action='store_true')

    args = parser.parse_args()

    run_detectree2_evaluation(
        gt_gpkg_path=args.gt,
        pred_gpkg_path=args.pred,
        output_path=args.output,
        iou_threshold=args.iou,
        conf_threshold=args.conf,
        area_threshold=args.area,
        clip_to_gt=not args.no_clip
    )


if __name__ == '__main__':
    main()
