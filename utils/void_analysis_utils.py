"""
Fonctions utilitaires V5
- Filtrage géométrique renforcé (voids = ronds et compacts)
- Post-traitement morphologique
- Seuil adaptatif amélioré
"""

import numpy as np
import cv2
from skimage import measure, morphology
from typing import Tuple, Dict, Optional


# ─── Prétraitement ────────────────────────────────────────────────────────────

def preprocess_image(image, adjust_contrast=1.0, adjust_brightness=0):
    adjusted = cv2.convertScaleAbs(image, alpha=adjust_contrast, beta=adjust_brightness)
    return cv2.bilateralFilter(adjusted, 9, 75, 75)


# ─── Masque ───────────────────────────────────────────────────────────────────

def apply_mask(image, mask):
    H, W = image.shape[:2]
    if mask.ndim == 3:
        g, r, b = mask[:,:,1], mask[:,:,2], mask[:,:,0]
        binary = ((g > 100) & (r < 100) & (b < 100)).astype(np.uint8)
    else:
        binary = (mask > 127).astype(np.uint8)
    if binary.shape != (H, W):
        binary = cv2.resize(binary, (W, H), interpolation=cv2.INTER_NEAREST)
        binary = (binary > 0).astype(np.uint8)
    if image.ndim == 2:
        return cv2.bitwise_and(image, image, mask=binary), binary
    masked = image.copy(); masked[binary == 0] = 0
    return masked, binary


# ─── Resize avec conservation du ratio ────────────────────────────────────────

def resize_with_aspect_ratio(image, target_size, pad_color=0):
    h, w = image.shape[:2]
    TH, TW = target_size
    scale = min(TW / w, TH / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_top    = (TH - nh) // 2
    pad_bottom = TH - nh - pad_top
    pad_left   = (TW - nw) // 2
    pad_right  = TW - nw - pad_left
    val = [pad_color]*3 if image.ndim == 3 else pad_color
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=val)
    if padded.shape[0] != TH or padded.shape[1] != TW:
        padded = padded[:TH, :TW] if image.ndim == 2 else padded[:TH, :TW, :]
        pb = max(0, TH - padded.shape[0]); pr = max(0, TW - padded.shape[1])
        if pb or pr:
            padded = cv2.copyMakeBorder(padded, 0, pb, 0, pr,
                                        cv2.BORDER_CONSTANT, value=val)
    transform = dict(scale=scale, pad_top=pad_top, pad_left=pad_left,
                     nh=nh, nw=nw, orig_h=h, orig_w=w)
    return padded, transform


def remove_padding_and_restore(pred_padded, transform):
    pt, pl = transform["pad_top"], transform["pad_left"]
    nh, nw = transform["nh"], transform["nw"]
    oh, ow = transform["orig_h"], transform["orig_w"]
    hp, wp = pred_padded.shape[:2]
    r1 = min(pt + nh, hp); c1 = min(pl + nw, wp)
    cropped = pred_padded[pt:r1, pl:c1] if pred_padded.ndim == 2 \
              else pred_padded[pt:r1, pl:c1, :]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return cv2.resize(pred_padded, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_LINEAR)


# ─── Filtrage géométrique renforcé ────────────────────────────────────────────

def filter_geometric_shapes(binary_mask):
    """
    Conserve uniquement les blobs ayant la morphologie d'un void de soudure :
    - Compact (pas trop allongé : ratio axes > 0.25)
    - Pas trop grand (< 30% de la surface du masque)
    - Pas un rectangle parfait (bord de composant)
    - Pas un cercle parfait ET très grand (via de fixation)
    """
    labeled = measure.label(binary_mask, connectivity=2)
    total   = binary_mask.shape[0] * binary_mask.shape[1]
    regions = measure.regionprops(labeled)
    filtered = np.zeros_like(binary_mask)

    for r in regions:
        if r.perimeter == 0 or r.major_axis_length == 0:
            continue
        circ = (4 * np.pi * r.area) / (r.perimeter ** 2)
        ar   = r.minor_axis_length / r.major_axis_length   # 0=allongé, 1=rond
        ext  = r.area / r.bbox_area if r.bbox_area > 0 else 0
        size_ratio = r.area / total

        # Exclure formes trop allongées (bords de composant, pistes)
        if ar < 0.25:
            continue
        # Exclure grands rectangles (bord de composant = rectangle ext>0.88)
        if ext > 0.88 and ar < 0.55:
            continue
        # Exclure très grands blobs (> 25% surface totale = fond/bord)
        if size_ratio > 0.25:
            continue
        # Exclure vias de fixation : cercle parfait ET grand
        if circ > 0.90 and ar > 0.88 and r.area > total * 0.01:
            continue

        filtered[labeled == r.label] = 1

    return filtered


# ─── Post-traitement morphologique ────────────────────────────────────────────

def morphological_cleanup(void_mask, min_area=50):
    """
    Nettoie le masque de voids :
    1. Fermeture pour combler les petits trous
    2. Ouverture pour enlever les pixels isolés
    3. Suppression des blobs trop petits
    """
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    closed = cv2.morphologyEx(void_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,  kernel_open)

    # Supprimer les blobs < min_area px
    labeled  = measure.label(opened, connectivity=2)
    filtered = np.zeros_like(opened)
    for r in measure.regionprops(labeled):
        if r.area >= min_area:
            filtered[labeled == r.label] = 1

    return filtered


# ─── Analyse des voids ────────────────────────────────────────────────────────

def analyze_voids(prediction, inspection_mask, filter_geometric=True,
                  void_threshold=None):
    """
    Seuil adaptatif + post-traitement morphologique + filtrage géométrique.
    """
    void_prob = prediction[:, :, 1]
    roi_vals  = void_prob[inspection_mask > 0]

    # Seuil adaptatif : percentile 80 dans la ROI, borné 0.03–0.50
    if void_threshold is None:
        if len(roi_vals) > 0 and roi_vals.max() > 0.005:
            thr = float(np.percentile(roi_vals, 80))
            thr = float(np.clip(thr, 0.03, 0.50))
        else:
            thr = 0.08
    else:
        thr = void_threshold

    soudure_mask = (prediction[:,:,0] > 0.5) & (inspection_mask > 0)
    voids_raw    = (void_prob > thr)          & (inspection_mask > 0)

    # Post-traitement morphologique
    voids_clean = morphological_cleanup(voids_raw.astype(np.uint8)) > 0

    # Filtrage géométrique
    if filter_geometric and voids_clean.any():
        voids_mask = filter_geometric_shapes(voids_clean.astype(np.uint8)) > 0
    else:
        voids_mask = voids_clean

    total     = int(np.sum(inspection_mask > 0))
    n_soudure = int(np.sum(soudure_mask))
    n_voids   = int(np.sum(voids_mask))
    void_ratio = n_voids / total * 100 if total > 0 else 0.0

    # Plus gros void intérieur
    lv_area=0; lv_ratio=0.0; lv_bbox=None; lv_centroid=None
    if voids_mask.any():
        labeled = measure.label(voids_mask.astype(np.uint8), connectivity=2)
        regions = measure.regionprops(labeled)
        interior = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox
            touches = (minr < 5 or minc < 5 or
                       maxr > inspection_mask.shape[0]-5 or
                       maxc > inspection_mask.shape[1]-5)
            if not touches:
                for y, x in r.coords[:20]:
                    if 0<y<inspection_mask.shape[0]-1 and 0<x<inspection_mask.shape[1]-1:
                        if np.any(inspection_mask[y-1:y+2, x-1:x+2] == 0):
                            touches = True; break
            if not touches:
                interior.append(r)
        if interior:
            lv = max(interior, key=lambda x: x.area)
            lv_area     = lv.area
            lv_ratio    = lv_area / total * 100 if total > 0 else 0.0
            lv_bbox     = lv.bbox
            lv_centroid = lv.centroid

    num_blobs = int(measure.label(voids_mask.astype(np.uint8)).max())

    return dict(
        void_ratio=float(void_ratio),
        largest_void_ratio=float(lv_ratio),
        largest_void_area=lv_area,
        largest_void_bbox=lv_bbox,
        largest_void_centroid=lv_centroid,
        num_voids=num_blobs,
        total_inspection_area=total,
        soudure_area=n_soudure,
        voids_area=n_voids,
        void_threshold_used=float(thr),
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

def create_visualization(original_image, prediction, inspection_mask,
                         analysis_results):
    """
    Visualisation claire :
    - Fond dans masque         → image originale assombrie 50%
    - Soudure (canal 0 > 0.5)  → teinte bleue subtile (on voit encore l'image)
    - Void (canal 1 > seuil)   → rouge vif plein opaque + contour blanc
    - Zones exclues            → noir absolu
    - Contour masque           → vert fin
    - Plus gros void           → cadre bleu ciel + croix
    """
    if original_image.ndim == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        base = original_image.copy()
    H, W = base.shape[:2]

    thr      = analysis_results.get("void_threshold_used", 0.08)
    vp       = prediction[:,:,1]
    soudure  = (prediction[:,:,0] > 0.5) & (inspection_mask > 0)
    voids    = (vp > thr)                & (inspection_mask > 0)
    fond     = (inspection_mask > 0)     & ~soudure & ~voids
    exclu    = (inspection_mask == 0)

    result = base.astype(np.float32).copy()

    # Zones exclues → noir absolu
    result[exclu] = 0

    # Fond → image assombrie 50%
    result[fond] = result[fond] * 0.50

    # Soudure → légère teinte bleue (garde la texture visible)
    result[soudure, 0] = result[soudure, 0] * 0.20
    result[soudure, 1] = result[soudure, 1] * 0.20
    result[soudure, 2] = np.clip(result[soudure, 2] * 0.40 + 140, 0, 220)

    # Voids → rouge vif 100% (plein, pas de mélange)
    if voids.any():
        result[voids, 0] = 240   # R
        result[voids, 1] = 20    # G
        result[voids, 2] = 20    # B

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Contour masque → vert fin
    cnts, _ = cv2.findContours(inspection_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0, 200, 0), 2)

    # Contours blancs autour de chaque void
    if voids.any():
        vc, _ = cv2.findContours(voids.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, vc, -1, (255, 255, 255), 1)

    # Cadre + croix du plus gros void → bleu ciel
    if analysis_results.get("largest_void_bbox") is not None:
        minr, minc, maxr, maxc = analysis_results["largest_void_bbox"]
        cv2.rectangle(result, (minc-3, minr-3), (maxc+3, maxr+3), (80, 220, 255), 3)
        if analysis_results.get("largest_void_centroid"):
            cy, cx = map(int, analysis_results["largest_void_centroid"])
            cv2.line(result, (cx-16,cy), (cx+16,cy), (80,220,255), 2)
            cv2.line(result, (cx,cy-16), (cx,cy+16), (80,220,255), 2)

    return result
