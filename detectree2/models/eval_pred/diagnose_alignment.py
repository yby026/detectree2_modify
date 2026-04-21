"""
Diagnostic script to check why IoU is near zero.
Checks spatial alignment between GT and predictions.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import box


def diagnose_alignment(gt_gpkg_path, pred_gpkg_path):
    """Check spatial alignment between GT and predictions."""
    
    print("="*60)
    print("DIAGNOSTIC: Checking spatial alignment")
    print("="*60)
    
    # Load data
    gt = gpd.read_file(gt_gpkg_path)
    pred = gpd.read_file(pred_gpkg_path)
    
    print(f"\n1. CRS Check:")
    print(f"   GT CRS:   {gt.crs}")
    print(f"   Pred CRS: {pred.crs}")
    print(f"   Match: {gt.crs == pred.crs}")
    
    print(f"\n2. Bounds Check:")
    gt_bounds = gt.total_bounds
    pred_bounds = pred.total_bounds
    print(f"   GT bounds:   [{gt_bounds[0]:.2f}, {gt_bounds[1]:.2f}, {gt_bounds[2]:.2f}, {gt_bounds[3]:.2f}]")
    print(f"   Pred bounds: [{pred_bounds[0]:.2f}, {pred_bounds[1]:.2f}, {pred_bounds[2]:.2f}, {pred_bounds[3]:.2f}]")
    
    # Check overlap
    gt_box = box(*gt_bounds)
    pred_box = box(*pred_bounds)
    
    if gt_box.intersects(pred_box):
        overlap = gt_box.intersection(pred_box)
        overlap_pct = (overlap.area / gt_box.area) * 100
        print(f"   Overlap: {overlap_pct:.1f}% of GT extent")
    else:
        print("   WARNING: NO SPATIAL OVERLAP!")
    
    print(f"\n3. Crown Count:")
    print(f"   GT crowns:   {len(gt)}")
    print(f"   Pred crowns: {len(pred)}")
    
    print(f"\n4. Crown Area Statistics (m²):")
    gt_areas = gt.geometry.area
    pred_areas = pred.geometry.area
    
    print(f"   GT   - min: {gt_areas.min():.1f}, median: {gt_areas.median():.1f}, max: {gt_areas.max():.1f}")
    print(f"   Pred - min: {pred_areas.min():.1f}, median: {pred_areas.median():.1f}, max: {pred_areas.max():.1f}")
    
    print(f"\n5. Sample Coordinates (first 3 crowns):")
    print("   GT centroids:")
    for i, row in gt.head(3).iterrows():
        c = row.geometry.centroid
        print(f"      {i}: ({c.x:.2f}, {c.y:.2f})")
    
    print("   Pred centroids:")
    for i, row in pred.head(3).iterrows():
        c = row.geometry.centroid
        print(f"      {i}: ({c.x:.2f}, {c.y:.2f})")
    
    # Clip predictions to GT extent
    print(f"\n6. Intersection Analysis:")
    gt_extent = box(*gt_bounds)
    pred_clipped = pred[pred.geometry.intersects(gt_extent)].copy()
    print(f"   Predictions within GT extent: {len(pred_clipped)}")
    
    # Check actual IoU between some pairs
    print(f"\n7. Sample IoU Check (first 10 GT crowns):")
    
    # Build spatial index
    pred_sindex = pred_clipped.sindex
    
    ious_found = []
    for idx, gt_row in gt.head(10).iterrows():
        gt_geom = gt_row.geometry
        candidates = list(pred_sindex.intersection(gt_geom.bounds))
        
        best_iou = 0
        for cand_idx in candidates:
            pred_geom = pred_clipped.iloc[cand_idx].geometry
            if gt_geom.intersects(pred_geom):
                try:
                    intersection = gt_geom.intersection(pred_geom).area
                    union = gt_geom.union(pred_geom).area
                    iou = intersection / union if union > 0 else 0
                    best_iou = max(best_iou, iou)
                except:
                    pass
        
        ious_found.append(best_iou)
        print(f"   GT crown {idx}: best IoU = {best_iou:.3f}")
    
    print(f"\n8. IoU Summary for sampled GT crowns:")
    ious_arr = np.array(ious_found)
    print(f"   Mean IoU:   {ious_arr.mean():.3f}")
    print(f"   Max IoU:    {ious_arr.max():.3f}")
    print(f"   IoU > 0.5:  {(ious_arr > 0.5).sum()} / {len(ious_arr)}")
    print(f"   IoU > 0.3:  {(ious_arr > 0.3).sum()} / {len(ious_arr)}")
    print(f"   IoU > 0.1:  {(ious_arr > 0.1).sum()} / {len(ious_arr)}")
    print(f"   IoU > 0.0:  {(ious_arr > 0.0).sum()} / {len(ious_arr)}")
    
    # Extended check: all GT crowns
    print(f"\n9. Full IoU Analysis (all GT crowns):")
    all_ious = []
    for idx, gt_row in gt.iterrows():
        gt_geom = gt_row.geometry
        candidates = list(pred_sindex.intersection(gt_geom.bounds))
        
        best_iou = 0
        for cand_idx in candidates:
            pred_geom = pred_clipped.iloc[cand_idx].geometry
            if gt_geom.intersects(pred_geom):
                try:
                    intersection = gt_geom.intersection(pred_geom).area
                    union = gt_geom.union(pred_geom).area
                    iou = intersection / union if union > 0 else 0
                    best_iou = max(best_iou, iou)
                except:
                    pass
        all_ious.append(best_iou)
    
    all_ious_arr = np.array(all_ious)
    print(f"   Total GT crowns: {len(all_ious_arr)}")
    print(f"   Mean IoU:   {all_ious_arr.mean():.3f}")
    print(f"   Median IoU: {np.median(all_ious_arr):.3f}")
    print(f"   Max IoU:    {all_ious_arr.max():.3f}")
    print(f"   IoU > 0.5:  {(all_ious_arr > 0.5).sum()} ({(all_ious_arr > 0.5).sum()/len(all_ious_arr)*100:.1f}%)")
    print(f"   IoU > 0.3:  {(all_ious_arr > 0.3).sum()} ({(all_ious_arr > 0.3).sum()/len(all_ious_arr)*100:.1f}%)")
    print(f"   IoU > 0.1:  {(all_ious_arr > 0.1).sum()} ({(all_ious_arr > 0.1).sum()/len(all_ious_arr)*100:.1f}%)")
    print(f"   IoU = 0:    {(all_ious_arr == 0).sum()} ({(all_ious_arr == 0).sum()/len(all_ious_arr)*100:.1f}%)")
    
    return {
        'gt_bounds': gt_bounds,
        'pred_bounds': pred_bounds,
        'all_ious': all_ious_arr
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        diagnose_alignment(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python diagnose_alignment.py <gt.gpkg> <pred.gpkg>")
