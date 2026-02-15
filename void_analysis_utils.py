"""
Fonctions utilitaires V4 — padding correct, seuil adaptatif, visu améliorée.
"""

import numpy as np
import cv2
from skimage import measure
from typing import Tuple, Dict, Optional


# ─── Prétraitement ────────────────────────────────────────────────────────────

def preprocess_image(image: np.ndarray,
                     adjust_contrast: float = 1.0,
                     adjust_brightness: int = 0) -> np.ndarray:
    adjusted = cv2.convertScaleAbs(image, alpha=adjust_contrast, beta=adjust_brightness)
    return cv2.bilateralFilter(adjusted, 9, 75, 75)


# ─── Masque ───────────────────────────────────────────────────────────────────

def apply_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

def resize_with_aspect_ratio(image: np.ndarray,
                              target_size: Tuple[int, int],
                              pad_color: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    Resize + padding centré.
    target_size = (target_H, target_W)
    Retourne (image_paddée, transform_dict)
    transform_dict contient tout ce qu'il faut pour inverser proprement.
    """
    h, w   = image.shape[:2]
    TH, TW = target_size

    # Ratio qui conserve les proportions
    scale = min(TW / w, TH / h)
    nw    = int(round(w * scale))
    nh    = int(round(h * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Padding centré
    pad_top    = (TH - nh) // 2
    pad_bottom = TH - nh - pad_top
    pad_left   = (TW - nw) // 2
    pad_right  = TW - nw - pad_left

    val = [pad_color] * 3 if image.ndim == 3 else pad_color
    padded = cv2.copyMakeBorder(resized,
                                pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=val)

    # Forcer la taille exacte (arrondi entier peut donner ±1 px)
    if padded.shape[0] != TH or padded.shape[1] != TW:
        if image.ndim == 3:
            padded = padded[:TH, :TW, :]
        else:
            padded = padded[:TH, :TW]
        # Recalcul précis
        pad_bottom = max(0, TH - padded.shape[0])
        pad_right  = max(0, TW - padded.shape[1])
        if pad_bottom or pad_right:
            padded = cv2.copyMakeBorder(padded, 0, pad_bottom, 0, pad_right,
                                        cv2.BORDER_CONSTANT, value=val)

    transform = dict(scale=scale,
                     pad_top=pad_top, pad_left=pad_left,
                     nh=nh, nw=nw,
                     orig_h=h, orig_w=w)
    return padded, transform


def remove_padding_and_restore(pred_padded: np.ndarray,
                                transform: Dict) -> np.ndarray:
    """
    1. Découpe la zone utile (enlève le padding)
    2. Resize vers la taille originale de l'image
    → Pas de distorsion car on utilise exactement le même ratio
    """
    pt = transform["pad_top"]
    pl = transform["pad_left"]
    nh = transform["nh"]
    nw = transform["nw"]
    orig_h = transform["orig_h"]
    orig_w = transform["orig_w"]

    # Sécurité : clip aux bords
    h_pred, w_pred = pred_padded.shape[:2]
    r1 = min(pt + nh, h_pred)
    c1 = min(pl + nw, w_pred)

    if pred_padded.ndim == 3:
        cropped = pred_padded[pt:r1, pl:c1, :]
    else:
        cropped = pred_padded[pt:r1, pl:c1]

    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        # Fallback si le crop est vide (ne devrait pas arriver)
        return cv2.resize(pred_padded, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


# ─── Filtrage géométrique ─────────────────────────────────────────────────────

def filter_geometric_shapes(binary_mask: np.ndarray) -> np.ndarray:
    labeled  = measure.label(binary_mask, connectivity=2)
    regions  = measure.regionprops(labeled)
    filtered = np.zeros_like(binary_mask)
    for r in regions:
        if r.perimeter == 0 or r.major_axis_length == 0:
            continue
        circ = (4 * np.pi * r.area) / (r.perimeter ** 2)
        ar   = r.minor_axis_length / r.major_axis_length
        ext  = r.area / r.bbox_area if r.bbox_area > 0 else 0
        # Exclure cercles parfaits, rectangles parfaits, formes allongées
        if not ((circ > 0.92 and ar > 0.90) or
                (ext  > 0.92 and ar < 0.45) or
                (ar   < 0.20)):
            filtered[labeled == r.label] = 1
    return filtered


# ─── Analyse des voids ────────────────────────────────────────────────────────

def analyze_voids(prediction: np.ndarray,
                  inspection_mask: np.ndarray,
                  filter_geometric: bool = True,
                  void_threshold: Optional[float] = None) -> Dict:
    """
    Seuil adaptatif : percentile 85 des probas void dans la ROI.
    Borné entre 0.03 et 0.55 pour couvrir les modèles peu confiants.
    """
    void_prob = prediction[:, :, 1]
    roi_vals  = void_prob[inspection_mask > 0]

    if void_threshold is None:
        if len(roi_vals) > 0 and roi_vals.max() > 0.005:
            thr = float(np.percentile(roi_vals, 85))
            thr = float(np.clip(thr, 0.03, 0.55))
        else:
            thr = 0.10
    else:
        thr = void_threshold

    soudure_mask = (prediction[:,:,0] > 0.5) & (inspection_mask > 0)
    voids_raw    = (void_prob > thr)          & (inspection_mask > 0)

    if filter_geometric and voids_raw.any():
        voids_mask = filter_geometric_shapes(voids_raw.astype(np.uint8)) > 0
    else:
        voids_mask = voids_raw

    total = int(np.sum(inspection_mask > 0))
    n_soudure = int(np.sum(soudure_mask))
    n_voids   = int(np.sum(voids_mask))
    void_ratio = n_voids / total * 100 if total > 0 else 0.0

    # Plus gros void intérieur
    lv_area = 0; lv_ratio = 0.0; lv_bbox = None; lv_centroid = None

    if voids_mask.any():
        labeled  = measure.label(voids_mask.astype(np.uint8), connectivity=2)
        regions  = measure.regionprops(labeled)
        interior = []
        for r in regions:
            minr, minc, maxr, maxc = r.bbox
            touches = (minr < 3 or minc < 3 or
                       maxr > inspection_mask.shape[0]-3 or
                       maxc > inspection_mask.shape[1]-3)
            if not touches:
                for y, x in r.coords[:30]:
                    if 0 < y < inspection_mask.shape[0]-1 and \
                       0 < x < inspection_mask.shape[1]-1:
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

    return dict(void_ratio=float(void_ratio),
                largest_void_ratio=float(lv_ratio),
                largest_void_area=lv_area,
                largest_void_bbox=lv_bbox,
                largest_void_centroid=lv_centroid,
                num_voids=num_blobs,
                total_inspection_area=total,
                soudure_area=n_soudure,
                voids_area=n_voids,
                void_threshold_used=float(thr))


# ─── Visualisation ────────────────────────────────────────────────────────────

def create_visualization(original_image: np.ndarray,
                         prediction: np.ndarray,
                         inspection_mask: np.ndarray,
                         analysis_results: Dict) -> np.ndarray:
    """
    Image annotée avec couleurs nettes :
      Soudure  → bleu  (0, 50, 210)   overlay fort
      Void     → rouge (220, 0, 0)    overlay très fort + contour blanc
      Fond     → image assombrie
      Exclu    → noir absolu
      Cadre    → bleu ciel (80, 220, 255) + croix
      Contour masque → vert fin
    """
    if original_image.ndim == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        base = original_image.copy()
    H, W = base.shape[:2]

    thr       = analysis_results.get("void_threshold_used", 0.10)
    void_prob = prediction[:,:,1]
    soudure   = (prediction[:,:,0] > 0.5)  & (inspection_mask > 0)
    voids     = (void_prob > thr)           & (inspection_mask > 0)
    fond      = (inspection_mask > 0)      & ~soudure & ~voids
    exclu     = (inspection_mask == 0)

    result = base.astype(np.float32).copy()

    # Zones exclues → noir
    result[exclu] = 0

    # Fond dans masque → assombri 40%
    result[fond]  = result[fond] * 0.40

    # Soudure → bleu foncé, mélange 70%
    bleu = np.array([0, 50, 210], dtype=np.float32)
    result[soudure] = result[soudure] * 0.30 + bleu * 0.70

    # Voids → rouge vif, mélange proportionnel à la confiance
    if voids.any():
        # Intensité variable selon la probabilité (rendu plus expressif)
        conf = np.clip(void_prob[voids] / max(thr, 0.01), 1.0, 4.0)
        r_ch = np.clip(180 + 20*(conf-1), 180, 255)
        result[voids, 0] = r_ch          # R
        result[voids, 1] = result[voids, 1] * 0.05   # G quasi nul
        result[voids, 2] = result[voids, 2] * 0.05   # B quasi nul

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Contour du masque d'inspection → vert
    cnts, _ = cv2.findContours(inspection_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0, 200, 0), 2)

    # Contours fins blancs autour des voids détectés
    if voids.any():
        vcnts, _ = cv2.findContours(voids.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, vcnts, -1, (255, 200, 200), 1)

    # Cadre + croix du plus gros void → bleu ciel épais
    if analysis_results.get("largest_void_bbox") is not None:
        minr, minc, maxr, maxc = analysis_results["largest_void_bbox"]
        cv2.rectangle(result, (minc-2, minr-2), (maxc+2, maxr+2), (80, 220, 255), 3)
        if analysis_results.get("largest_void_centroid"):
            cy, cx = map(int, analysis_results["largest_void_centroid"])
            cv2.line(result, (cx-14,cy), (cx+14,cy), (80,220,255), 2)
            cv2.line(result, (cx,cy-14), (cx,cy+14), (80,220,255), 2)

    return result
