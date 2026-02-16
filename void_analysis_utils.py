"""
Utilitaires V6 — Approche hybride IA + seuillage local.

LOGIQUE :
  1. Le modèle IA identifie la ZONE DE SOUDURE (canal 0)
     → tâche simple, robuste même avec peu de données
  2. Dans cette zone, un seuillage adaptatif local sur l'image RX
     trouve les VOIDS (les zones les plus sombres = absence de métal)
     → physiquement fiable, indépendant de l'entraînement void

RENDU FINAL (3 zones dans le masque) :
  🟢 Vert        — Soudure présente (zone soudure sans void)
  🔴 Rouge vif   — Void / manque de soudure
  ⬛ Noir        — Zone exclue par le masque
"""

import numpy as np
import cv2
from skimage import measure
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
    nw = int(round(w * scale)); nh = int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_top    = (TH - nh) // 2; pad_bottom = TH - nh - pad_top
    pad_left   = (TW - nw) // 2; pad_right  = TW - nw - pad_left
    val = [pad_color]*3 if image.ndim == 3 else pad_color
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=val)
    if padded.shape[0] != TH or padded.shape[1] != TW:
        padded = padded[:TH, :TW] if image.ndim == 2 else padded[:TH, :TW, :]
        pb = max(0, TH-padded.shape[0]); pr = max(0, TW-padded.shape[1])
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


# ─── Détection zone soudure (IA) ──────────────────────────────────────────────

def detect_solder_zone(prediction, inspection_mask, solder_threshold=None):
    """
    Extrait la zone de soudure depuis la prédiction IA (canal 0).
    Seuil adaptatif : percentile 40 des proba canal 0 dans le masque
    (on veut inclure une bonne partie du masque comme soudure).
    """
    solder_prob = prediction[:, :, 0]
    roi = solder_prob[inspection_mask > 0]

    if solder_threshold is None:
        if len(roi) > 0 and roi.max() > 0.01:
            thr = float(np.percentile(roi, 40))
            thr = float(np.clip(thr, 0.10, 0.70))
        else:
            thr = 0.30
    else:
        thr = solder_threshold

    zone = (solder_prob > thr) & (inspection_mask > 0)

    # Fermeture morphologique pour combler les trous
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    zone = cv2.morphologyEx(zone.astype(np.uint8),
                            cv2.MORPH_CLOSE, k).astype(bool)
    # Conserver uniquement dans le masque d'inspection
    zone = zone & (inspection_mask > 0)
    return zone, float(thr)


# ─── Détection voids par seuillage local (physique) ──────────────────────────

def detect_voids_threshold(gray_image, solder_zone_mask,
                           sensitivity=50, min_void_px=30):
    """
    Détecte les voids par seuillage adaptatif LOCAL dans la zone soudure.

    Principe physique : dans une image RX, la soudure est grise/sombre,
    les voids (absence de métal) sont encore plus sombres car moins dense.
    → Les voids sont les pixels les plus SOMBRES dans la zone soudure.

    Args:
        gray_image     : image RX en niveaux de gris (uint8)
        solder_zone_mask: masque booléen de la zone soudure
        sensitivity    : percentile de coupure (0–100).
                         50 = pixels en-dessous de la médiane locale → void.
                         Plus élevé = plus de voids détectés.
        min_void_px    : surface minimum d'un void en pixels

    Returns:
        void_mask (bool), threshold_used (float)
    """
    if not solder_zone_mask.any():
        return np.zeros_like(solder_zone_mask), 0.0

    # Valeurs de gris dans la zone soudure uniquement
    vals = gray_image[solder_zone_mask]

    # Seuil = percentile (sensitivity) des valeurs dans la soudure
    # → les pixels les plus sombres de la soudure = voids
    thr = float(np.percentile(vals, sensitivity))

    # Masque : pixels sombres dans la zone soudure
    void_raw = (gray_image.astype(np.float32) < thr) & solder_zone_mask

    # Nettoyage morphologique
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(void_raw.astype(np.uint8), cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned,                   cv2.MORPH_CLOSE, k_close)

    # Supprimer les micro-blobs
    labeled = measure.label(cleaned, connectivity=2)
    filtered = np.zeros_like(cleaned)
    for r in measure.regionprops(labeled):
        if r.area >= min_void_px:
            filtered[labeled == r.label] = 1

    return filtered.astype(bool), thr


# ─── Analyse ──────────────────────────────────────────────────────────────────

def analyze_voids(prediction, inspection_mask,
                  filter_geometric=True,
                  void_threshold=None,
                  gray_image=None,
                  sensitivity=50,
                  solder_threshold=None):
    """
    Analyse hybride :
      - Zone soudure  → IA (canal 0)
      - Voids         → seuillage local sur l'image RX

    Args:
        prediction       : sortie modèle (H,W,3)
        inspection_mask  : masque binaire zone inspectée
        gray_image       : image RX en gris (nécessaire pour le seuillage)
        sensitivity      : percentile seuillage void (20–60 recommandé)
        solder_threshold : seuil IA zone soudure (None = adaptatif)
        void_threshold   : ignoré (conservé pour compatibilité)
        filter_geometric : non utilisé (le seuillage physique est déjà propre)
    """
    total = int(np.sum(inspection_mask > 0))

    # ── 1. Zone soudure via IA ────────────────────────────────────────────────
    solder_zone, solder_thr = detect_solder_zone(
        prediction, inspection_mask, solder_threshold)

    # ── 2. Voids via seuillage physique ───────────────────────────────────────
    if gray_image is not None and solder_zone.any():
        void_mask, void_thr = detect_voids_threshold(
            gray_image, solder_zone, sensitivity=sensitivity)
    else:
        # Fallback IA si pas d'image grise
        vp  = prediction[:, :, 1]
        roi = vp[inspection_mask > 0]
        thr = float(np.clip(np.percentile(roi, 80), 0.03, 0.50)) \
              if len(roi) > 0 and roi.max() > 0.005 else 0.10
        void_mask = (vp > thr) & solder_zone
        void_thr  = thr

    # La soudure PRÉSENTE = zone soudure MOINS les voids
    solder_present = solder_zone & ~void_mask

    # ── 3. Métriques ─────────────────────────────────────────────────────────
    n_solder = int(np.sum(solder_zone))
    n_voids  = int(np.sum(void_mask))
    void_ratio = n_voids / n_solder * 100 if n_solder > 0 else 0.0

    # Plus gros void intérieur (sans contact bord masque)
    lv_area=0; lv_ratio=0.0; lv_bbox=None; lv_centroid=None
    if void_mask.any():
        labeled = measure.label(void_mask.astype(np.uint8), connectivity=2)
        interior = []
        for r in measure.regionprops(labeled):
            mr, mc, xr, xc = r.bbox
            if (mr < 5 or mc < 5 or
                    xr > inspection_mask.shape[0]-5 or
                    xc > inspection_mask.shape[1]-5):
                continue
            touch = False
            for y, x in r.coords[:20]:
                if 0<y<inspection_mask.shape[0]-1 and \
                   0<x<inspection_mask.shape[1]-1:
                    if np.any(inspection_mask[y-1:y+2, x-1:x+2] == 0):
                        touch = True; break
            if not touch:
                interior.append(r)
        if interior:
            lv = max(interior, key=lambda r: r.area)
            lv_area     = lv.area
            lv_ratio    = lv_area / n_solder * 100 if n_solder > 0 else 0.0
            lv_bbox     = lv.bbox
            lv_centroid = lv.centroid

    num_blobs = int(measure.label(void_mask.astype(np.uint8)).max())

    return dict(
        void_ratio=float(void_ratio),
        largest_void_ratio=float(lv_ratio),
        largest_void_area=lv_area,
        largest_void_bbox=lv_bbox,
        largest_void_centroid=lv_centroid,
        num_voids=num_blobs,
        total_inspection_area=total,
        solder_area=n_solder,
        voids_area=n_voids,
        solder_zone=solder_zone,       # booléen (H,W)
        void_mask=void_mask,           # booléen (H,W)
        void_threshold_used=float(void_thr),
        solder_threshold_used=float(solder_thr),
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

def create_visualization(original_image, prediction, inspection_mask,
                         analysis_results):
    """
    3 zones claires dans le masque d'inspection :

      ⬜ Gris clair (image originale légèrement éclaircie)
         → Soudure présente (zone soudure détectée SANS void)

      🔴 Rouge vif opaque
         → Void / manque de soudure

      ⬛ Noir absolu
         → Zone exclue par le masque (non analysée)

    + Contour vert du masque
    + Cadre bleu ciel + croix : plus gros void intérieur
    """
    if original_image.ndim == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        base = original_image.copy()
    H, W = base.shape[:2]

    solder_zone = analysis_results.get("solder_zone")
    void_mask   = analysis_results.get("void_mask")

    # Fallback si les masques ne sont pas dans les résultats
    if solder_zone is None:
        thr = analysis_results.get("void_threshold_used", 0.10)
        void_mask   = (prediction[:,:,1] > thr) & (inspection_mask > 0)
        solder_zone = (prediction[:,:,0] > 0.3)  & (inspection_mask > 0)

    solder_present = solder_zone & ~void_mask
    exclu          = inspection_mask == 0

    result = base.astype(np.float32).copy()

    # ── Zones exclues → noir absolu ───────────────────────────────────────────
    result[exclu] = 0

    # ── Zones dans le masque mais hors soudure → très sombre (fond neutre) ───
    hors_soudure = (inspection_mask > 0) & ~solder_zone
    result[hors_soudure] = result[hors_soudure] * 0.25

    # ── Soudure présente → vert saturé (avec texture image visible dessous)
    if solder_present.any():
        r_ch = result[solder_present, 0]
        g_ch = result[solder_present, 1]
        b_ch = result[solder_present, 2]
        result[solder_present, 0] = (r_ch * 0.15).clip(0, 255)          # R faible
        result[solder_present, 1] = (g_ch * 0.40 + 120).clip(0, 220)    # G dominant
        result[solder_present, 2] = (b_ch * 0.15).clip(0, 255)          # B faible

    # ── Voids → rouge vif 100% opaque ────────────────────────────────────────
    if void_mask is not None and void_mask.any():
        result[void_mask, 0] = 230   # R
        result[void_mask, 1] = 20    # G
        result[void_mask, 2] = 20    # B

    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Contour masque → vert ─────────────────────────────────────────────────
    cnts, _ = cv2.findContours(inspection_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0, 200, 0), 2)

    # ── Contours fins blancs autour des voids ─────────────────────────────────
    if void_mask is not None and void_mask.any():
        vc, _ = cv2.findContours(void_mask.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, vc, -1, (255, 255, 255), 1)

    # ── Cadre + croix plus gros void → bleu ciel ─────────────────────────────
    if analysis_results.get("largest_void_bbox") is not None:
        mr, mc, xr, xc = analysis_results["largest_void_bbox"]
        cv2.rectangle(result, (mc-3, mr-3), (xc+3, xr+3), (80, 220, 255), 3)
        if analysis_results.get("largest_void_centroid"):
            cy, cx = map(int, analysis_results["largest_void_centroid"])
            cv2.line(result, (cx-16,cy), (cx+16,cy), (80,220,255), 2)
            cv2.line(result, (cx,cy-16), (cx,cy+16), (80,220,255), 2)

    return result


# ─── Filtrage (conservé pour compatibilité imports) ───────────────────────────

def filter_geometric_shapes(binary_mask):
    labeled  = measure.label(binary_mask, connectivity=2)
    total    = binary_mask.shape[0] * binary_mask.shape[1]
    filtered = np.zeros_like(binary_mask)
    for r in measure.regionprops(labeled):
        if r.perimeter == 0 or r.major_axis_length == 0: continue
        ar  = r.minor_axis_length / r.major_axis_length
        ext = r.area / r.bbox_area if r.bbox_area > 0 else 0
        if not (ar < 0.25 or (ext > 0.88 and ar < 0.55) or
                r.area / total > 0.25):
            filtered[labeled == r.label] = 1
    return filtered
