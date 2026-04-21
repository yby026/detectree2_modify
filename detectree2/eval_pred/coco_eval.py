"""
COCO-style evaluation for tree crown detection.
Converts gpkg files to COCO format and runs standard AP evaluation.

Fixed version: properly handles geographic coordinates and COCO format requirements.
"""

import argparse
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate, scale
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util
import tempfile
import os


def polygon_to_rle(geom, img_height, img_width):
    """
    Convert shapely polygon to COCO RLE format.
    This is more robust than polygon coordinates for pycocotools.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Create a binary mask
    mask = Image.new('L', (int(img_width), int(img_height)), 0)
    draw = ImageDraw.Draw(mask)
    
    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
        # PIL expects [(x, y), (x, y), ...]
        draw.polygon(coords, fill=1)
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            draw.polygon(coords, fill=1)
    
    # Convert to numpy and then to RLE
    mask_arr = np.array(mask, dtype=np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask_arr))
    rle['counts'] = rle['counts'].decode('utf-8')  # Make JSON serializable
    
    return rle


def polygon_to_coco_segmentation(geom):
    """Convert shapely polygon to COCO polygon segmentation format."""
    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
        # COCO format: [x1, y1, x2, y2, x3, y3, ...]
        flat = []
        for x, y in coords:
            flat.extend([float(x), float(y)])
        return [flat]
    elif geom.geom_type == 'MultiPolygon':
        # Return all polygons
        result = []
        for poly in geom.geoms:
            coords = list(poly.exterior.coords)
            flat = []
            for x, y in coords:
                flat.extend([float(x), float(y)])
            result.append(flat)
        return result
    else:
        return None


def run_coco_evaluation(gt_gpkg_path, pred_gpkg_path, output_path=None,
                        area_thresholds=None, clip_to_gt=True, use_rle=False):
    """
    Run COCO-style evaluation on crown detection results.
    """
    print("Loading ground truth...")
    gt_gdf = gpd.read_file(gt_gpkg_path)
    print(f"  {len(gt_gdf)} ground truth crowns")
    print(f"  CRS: {gt_gdf.crs}")
    gt_bounds = gt_gdf.total_bounds
    print(f"  Bounds: {gt_bounds}")
    
    print("\nLoading predictions...")
    pred_gdf = gpd.read_file(pred_gpkg_path)
    print(f"  {len(pred_gdf)} predicted crowns (before filtering)")
    
    # Find confidence score column
    score_col = None
    for col in ['Confidence_score', 'score', 'confidence', 'Score']:
        if col in pred_gdf.columns:
            score_col = col
            print(f"  Using '{col}' as confidence score")
            print(f"  Score range: {pred_gdf[col].min():.3f} - {pred_gdf[col].max():.3f}")
            break
    
    if score_col is None:
        print("  WARNING: No confidence score column found!")
    
    # Clip predictions to GT extent
    if clip_to_gt:
        print("\nClipping predictions to GT extent...")
        gt_bbox = box(*gt_bounds)
        pred_gdf = pred_gdf[pred_gdf.geometry.intersects(gt_bbox)].copy()
        print(f"  {len(pred_gdf)} predictions within GT extent")
    
    # Calculate image dimensions and origin
    min_x, min_y = gt_bounds[0], gt_bounds[1]
    max_x, max_y = gt_bounds[2], gt_bounds[3]
    
    img_width = max_x - min_x
    img_height = max_y - min_y
    
    print(f"\nImage dimensions: {img_width:.1f} x {img_height:.1f} m")
    
    # Transform coordinates: shift to origin and flip Y axis for image coordinates
    print("Transforming to image coordinates...")
    gt_gdf = gt_gdf.copy()
    pred_gdf = pred_gdf.copy()
    
    # Shift to origin
    gt_gdf['geometry'] = gt_gdf.geometry.translate(-min_x, -min_y)
    pred_gdf['geometry'] = pred_gdf.geometry.translate(-min_x, -min_y)
    
    # Flip Y axis (geographic Y up -> image Y down)
    gt_gdf['geometry'] = gt_gdf.geometry.scale(xfact=1, yfact=-1, origin=(0, img_height/2))
    pred_gdf['geometry'] = pred_gdf.geometry.scale(xfact=1, yfact=-1, origin=(0, img_height/2))
    
    # Build annotations
    print("\nConverting to COCO format...")
    
    gt_anns = []
    for idx, (_, row) in enumerate(gt_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        bounds = geom.bounds
        bbox = [float(bounds[0]), float(bounds[1]),
                float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1])]
        
        segmentation = polygon_to_coco_segmentation(geom)
        if segmentation is None:
            continue
        
        gt_anns.append({
            'id': idx + 1,
            'image_id': 1,
            'category_id': 1,
            'bbox': bbox,
            'area': float(geom.area),
            'segmentation': segmentation,
            'iscrowd': 0
        })
    
    pred_anns = []
    for idx, (_, row) in enumerate(pred_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        
        bounds = geom.bounds
        bbox = [float(bounds[0]), float(bounds[1]),
                float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1])]
        
        segmentation = polygon_to_coco_segmentation(geom)
        if segmentation is None:
            continue
        
        score = 1.0
        if score_col is not None:
            score = float(row[score_col])
        
        pred_anns.append({
            'id': idx + 1,
            'image_id': 1,
            'category_id': 1,
            'bbox': bbox,
            'area': float(geom.area),
            'segmentation': segmentation,
            'score': score
        })
    
    print(f"  {len(gt_anns)} GT annotations")
    print(f"  {len(pred_anns)} prediction annotations")
    
    # Verify some annotations
    print("\nSample GT annotation bbox:", gt_anns[0]['bbox'] if gt_anns else "None")
    print("Sample Pred annotation bbox:", pred_anns[0]['bbox'] if pred_anns else "None")
    
    # Build COCO GT structure
    coco_gt_dict = {
        'info': {'description': 'Tree crown evaluation', 'version': '1.0', 'year': 2025},
        'licenses': [{'id': 1, 'name': 'Unknown', 'url': ''}],
        'images': [{
            'id': 1,
            'width': int(np.ceil(img_width)),
            'height': int(np.ceil(img_height)),
            'file_name': 'merged.tif'
        }],
        'categories': [{'id': 1, 'name': 'tree_crown', 'supercategory': 'vegetation'}],
        'annotations': gt_anns
    }
    
    # Save GT to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_gt_dict, f)
        gt_json_path = f.name
    
    # Also save pred to temp file (alternative loading method)
    pred_results = pred_anns  # Already in correct format for loadRes
    
    # Run evaluation
    print("\n" + "="*60)
    print("Running COCO evaluation (segmentation)...")
    print("="*60)
    
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_results)
    
    coco_eval_segm = COCOeval(coco_gt, coco_dt, 'segm')
    
    if area_thresholds is not None:
        coco_eval_segm.params.areaRng = [
            [0, 1e10],
            [0, area_thresholds[0]],
            [area_thresholds[0], area_thresholds[1]],
            [area_thresholds[1], 1e10]
        ]
        coco_eval_segm.params.areaRngLbl = ['all', 'small', 'medium', 'large']
    
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    segm_stats = coco_eval_segm.stats
    
    print("\n" + "="*60)
    print("Running COCO evaluation (bounding box)...")
    print("="*60)
    
    coco_eval_bbox = COCOeval(coco_gt, coco_dt, 'bbox')
    if area_thresholds is not None:
        coco_eval_bbox.params.areaRng = coco_eval_segm.params.areaRng
        coco_eval_bbox.params.areaRngLbl = coco_eval_segm.params.areaRngLbl
    
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    bbox_stats = coco_eval_bbox.stats
    
    # Clean up
    os.unlink(gt_json_path)
    
    # Compile results
    results = {
        'segmentation': {
            'AP': float(segm_stats[0]),
            'AP50': float(segm_stats[1]),
            'AP75': float(segm_stats[2]),
            'AP_small': float(segm_stats[3]),
            'AP_medium': float(segm_stats[4]),
            'AP_large': float(segm_stats[5]),
            'AR_max1': float(segm_stats[6]),
            'AR_max10': float(segm_stats[7]),
            'AR_max100': float(segm_stats[8]),
            'AR_small': float(segm_stats[9]),
            'AR_medium': float(segm_stats[10]),
            'AR_large': float(segm_stats[11]),
        },
        'bbox': {
            'AP': float(bbox_stats[0]),
            'AP50': float(bbox_stats[1]),
            'AP75': float(bbox_stats[2]),
            'AP_small': float(bbox_stats[3]),
            'AP_medium': float(bbox_stats[4]),
            'AP_large': float(bbox_stats[5]),
            'AR_max1': float(bbox_stats[6]),
            'AR_max10': float(bbox_stats[7]),
            'AR_max100': float(bbox_stats[8]),
            'AR_small': float(bbox_stats[9]),
            'AR_medium': float(bbox_stats[10]),
            'AR_large': float(bbox_stats[11]),
        },
        'metadata': {
            'n_gt': len(gt_anns),
            'n_pred': len(pred_anns),
            'gt_file': gt_gpkg_path,
            'pred_file': pred_gpkg_path,
            'area_thresholds': area_thresholds,
            'img_width': img_width,
            'img_height': img_height
        }
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nGround truth crowns: {len(gt_anns)}")
    print(f"Predicted crowns: {len(pred_anns)}")
    print(f"\nSegmentation metrics:")
    print(f"  AP (IoU=0.50:0.95): {results['segmentation']['AP']:.3f}")
    print(f"  AP50:               {results['segmentation']['AP50']:.3f}")
    print(f"  AP75:               {results['segmentation']['AP75']:.3f}")
    print(f"\nBounding box metrics:")
    print(f"  AP (IoU=0.50:0.95): {results['bbox']['AP']:.3f}")
    print(f"  AP50:               {results['bbox']['AP50']:.3f}")
    print(f"  AP75:               {results['bbox']['AP75']:.3f}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='COCO-style evaluation for tree crowns')
    parser.add_argument('--gt', required=True, help='Path to ground truth gpkg')
    parser.add_argument('--pred', required=True, help='Path to prediction gpkg')
    parser.add_argument('--output', default='coco_eval_results.json', help='Output JSON')
    parser.add_argument('--area-small', type=float, default=None)
    parser.add_argument('--area-large', type=float, default=None)
    parser.add_argument('--no-clip', action='store_true')
    
    args = parser.parse_args()
    
    area_thresholds = None
    if args.area_small is not None and args.area_large is not None:
        area_thresholds = [args.area_small, args.area_large]
    
    run_coco_evaluation(
        gt_gpkg_path=args.gt,
        pred_gpkg_path=args.pred,
        output_path=args.output,
        area_thresholds=area_thresholds,
        clip_to_gt=not args.no_clip
    )


if __name__ == '__main__':
    main()
