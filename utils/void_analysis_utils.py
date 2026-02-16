"""
Utilitaires V7 — Approche SANS IA, 100% seuillage physique.

PRINCIPE :
  L'utilisateur définit un masque vert = zone à inspecter.
  Dans cette zone, on applique un seuillage adaptatif local :
    - Les pixels CLAIRS  = voids (absence de métal = moins dense = plus de RX transmis)
    - Les pixels SOMBRES = soudure présente (métal dense = plus d'absorption)

  L'IA (si chargée) est utilisée UNIQUEMENT pour affiner la zone soudure
  à l'intérieur du masque — mais n'est pas requise.

RENDU (3 zones sans ambiguïté) :
  🟢 Vert  — Soudure présente
  🔴 Rouge — Void / manque
  ⬛ Noir  — Zone exclue par le masque
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


# ─── Seuillage adaptatif local ────────────────────────────────────────────────

def detect_voids_threshold(gray_image, roi_mask, sensitivity=35, min_void_px=60):
    """
    Détecte les voids par seuillage adaptatif LOCAL dans la ROI.

    PHYSIQUE RX : les voids sont les zones les plus CLAIRES.
    Absence de métal = moins dense = moins d'absorption RX = plus de rayons
    transmis = pixel plus clair sur le détecteur.

    Passe 1 — Seuil percentile HAUT dans la ROI
      On prend le percentile (100-sensitivity) des pixels dans le masque.
      Les pixels AU-DESSUS de ce seuil = candidats voids (zones claires).

    Passe 2 — Seuillage Otsu inversé
      Otsu trouve automatiquement la coupure soudure/void.

    Passe 3 — Combinaison : void = passe1 ET passe2 (double confirmation)

    Args:
        gray_image   : image RX prétraitée en niveaux de gris (uint8)
        roi_mask     : masque binaire uint8 (1 = zone à inspecter)
        sensitivity  : percentile (5–60). 35 = les 35% les plus CLAIRS = voids.
                       Augmenter → plus de voids détectés.
        min_void_px  : surface min d'un void en pixels (filtre le bruit)

    Returns:
        void_mask (bool H×W), seuil_utilisé (float)
    """
    if not roi_mask.any():
        return np.zeros(gray_image.shape, dtype=bool), 0.0

    vals = gray_image[roi_mask > 0]

    # PHYSIQUE RX : void = absence de métal = MOINS dense = PLUS CLAIR
    # Les voids sont les pixels les plus CLAIRS dans la zone soudure

    # ── Passe 1 : seuil percentile HAUT dans la ROI ───────────────────────────
    # sensitivity=35 → les 35% les plus CLAIRS sont des voids
    thr_global = float(np.percentile(vals, 100 - sensitivity))

    # ── Passe 2 : Otsu inversé ────────────────────────────────────────────────
    v_norm = ((vals - vals.min()) /
              max(vals.max() - vals.min(), 1) * 255).astype(np.uint8)
    thr_otsu_val, _ = cv2.threshold(v_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_otsu = float(vals.min() + thr_otsu_val / 255.0 * (vals.max() - vals.min()))

    # Seuil final = favorise le plus HAUT des deux (ne pas rater des voids clairs)
    thr_final = max(thr_global, thr_otsu * 0.9 + thr_global * 0.1)
    thr_final = float(np.clip(thr_final, vals.min() + 1, vals.max() - 1))

    # ── Application : pixels SUPÉRIEURS au seuil = voids (zones claires) ─────
    void_raw = (gray_image.astype(np.float32) >= thr_final) & (roi_mask > 0)

    # ── Nettoyage morphologique ───────────────────────────────────────────────
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(void_raw.astype(np.uint8), cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned,                   cv2.MORPH_CLOSE, k_close)

    # ── Suppression des micro-blobs ───────────────────────────────────────────
    labeled  = measure.label(cleaned, connectivity=2)
    filtered = np.zeros_like(cleaned)
    for r in measure.regionprops(labeled):
        if r.area >= min_void_px:
            filtered[labeled == r.label] = 1

    return filtered.astype(bool), thr_final


# ─── (Optionnel) Zone soudure via IA ─────────────────────────────────────────

def detect_solder_zone(prediction, inspection_mask, solder_threshold=None):
    """
    Raffine la zone d'inspection en utilisant le canal 0 du modèle IA.
    Si le modèle est peu fiable, retourne simplement le masque original.
    """
    if prediction is None:
        return (inspection_mask > 0), 0.0

    solder_prob = prediction[:, :, 0]
    roi = solder_prob[inspection_mask > 0]

    if len(roi) == 0 or roi.max() < 0.05:
        # Modèle peu confiant → utiliser tout le masque
        return (inspection_mask > 0), 0.0

    if solder_threshold is None:
        thr = float(np.clip(np.percentile(roi, 30), 0.05, 0.60))
    else:
        thr = solder_threshold

    zone = (solder_prob > thr) & (inspection_mask > 0)

    # Si l'IA couvre moins de 20% du masque → probablement peu fiable,
    # on revient au masque complet
    coverage = zone.sum() / max((inspection_mask > 0).sum(), 1)
    if coverage < 0.20:
        return (inspection_mask > 0), 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    zone = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    zone = zone & (inspection_mask > 0)
    return zone, float(thr)


# ─── Analyse principale ───────────────────────────────────────────────────────

def analyze_voids(prediction, inspection_mask,
                  filter_geometric=True,
                  void_threshold=None,
                  gray_image=None,
                  sensitivity=35,
                  min_void_px=60,
                  solder_threshold=None,
                  use_ai_zone=False):
    """
    Analyse des voids dans la zone d'inspection.

    Stratégie (par ordre de priorité) :
      1. Si gray_image fournie → seuillage physique direct dans le masque
         C'est la méthode principale, robuste, sans dépendance au modèle.
      2. use_ai_zone=True ET modèle confiant → raffine la zone via IA avant seuillage
      3. Fallback : canal 1 du modèle si pas d'image grise (mode legacy)

    Args:
        gray_image   : image RX prétraitée (REQUIS pour le seuillage physique)
        sensitivity  : percentile de coupure dans la ROI (5–60, défaut 35)
        min_void_px  : surface min void en pixels
        use_ai_zone  : utiliser l'IA pour affiner la zone soudure (expérimental)
    """
    total = int(np.sum(inspection_mask > 0))

    # ── Déterminer la zone de travail ─────────────────────────────────────────
    if use_ai_zone and prediction is not None:
        work_zone, solder_thr = detect_solder_zone(
            prediction, inspection_mask, solder_threshold)
    else:
        work_zone = (inspection_mask > 0)
        solder_thr = 0.0

    # ── Seuillage physique (méthode principale) ───────────────────────────────
    if gray_image is not None:
        void_mask, void_thr = detect_voids_threshold(
            gray_image, work_zone.astype(np.uint8),
            sensitivity=sensitivity,
            min_void_px=min_void_px)
    else:
        # Fallback IA (mode legacy si pas d'image grise)
        if prediction is not None:
            vp  = prediction[:, :, 1]
            roi = vp[inspection_mask > 0]
            thr = float(np.clip(np.percentile(roi, 80), 0.03, 0.50)) \
                  if len(roi) > 0 and roi.max() > 0.005 else 0.10
            void_mask = (vp > thr) & (inspection_mask > 0)
            void_thr  = thr
        else:
            void_mask = np.zeros(inspection_mask.shape, dtype=bool)
            void_thr  = 0.0

    # Soudure présente = zone travail SANS voids
    solder_present = work_zone & ~void_mask

    # ── Métriques ─────────────────────────────────────────────────────────────
    n_work   = int(np.sum(work_zone))
    n_voids  = int(np.sum(void_mask))
    void_ratio = n_voids / n_work * 100 if n_work > 0 else 0.0

    # Plus gros void intérieur (ne touche pas le bord du masque)
    lv_area=0; lv_ratio=0.0; lv_bbox=None; lv_centroid=None
    if void_mask.any():
        labeled  = measure.label(void_mask.astype(np.uint8), connectivity=2)
        interior = []
        for r in measure.regionprops(labeled):
            mr, mc, xr, xc = r.bbox
            if (mr < 5 or mc < 5 or
                    xr > inspection_mask.shape[0]-5 or
                    xc > inspection_mask.shape[1]-5):
                continue
            touch = False
            for y, x in r.coords[:20]:
                if (0 < y < inspection_mask.shape[0]-1 and
                        0 < x < inspection_mask.shape[1]-1):
                    if np.any(inspection_mask[y-1:y+2, x-1:x+2] == 0):
                        touch = True; break
            if not touch:
                interior.append(r)
        if interior:
            lv = max(interior, key=lambda r: r.area)
            lv_area     = lv.area
            lv_ratio    = lv_area / n_work * 100 if n_work > 0 else 0.0
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
        solder_area=n_work,
        voids_area=n_voids,
        solder_zone=work_zone,
        void_mask=void_mask,
        void_threshold_used=float(void_thr),
        solder_threshold_used=float(solder_thr),
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

def create_visualization(original_image, prediction, inspection_mask,
                         analysis_results):
    """
    Rendu 3 zones :
      🟢 Vert  → Soudure présente
      🔴 Rouge → Void / manque
      ⬛ Noir  → Zone exclue
    """
    if original_image.ndim == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        base = original_image.copy()

    solder_zone = analysis_results.get("solder_zone")
    void_mask   = analysis_results.get("void_mask")

    # Fallback si les masques ne sont pas dans les résultats
    if solder_zone is None:
        solder_zone = (inspection_mask > 0)
    if void_mask is None:
        void_mask = np.zeros(inspection_mask.shape, dtype=bool)

    solder_present = solder_zone & ~void_mask
    exclu          = (inspection_mask == 0)

    result = base.astype(np.float32).copy()

    # ── Noir absolu : zones exclues ───────────────────────────────────────────
    result[exclu] = 0

    # ── Vert : soudure présente (avec texture de l'image visible) ─────────────
    if solder_present.any():
        result[solder_present, 0] = (result[solder_present, 0] * 0.10).clip(0, 80)
        result[solder_present, 1] = (result[solder_present, 1] * 0.35 + 110).clip(0, 210)
        result[solder_present, 2] = (result[solder_present, 2] * 0.10).clip(0, 80)

    # ── Rouge vif : voids ─────────────────────────────────────────────────────
    if void_mask.any():
        result[void_mask, 0] = 235
        result[void_mask, 1] = 15
        result[void_mask, 2] = 15

    result = np.clip(result, 0, 255).astype(np.uint8)

    # ── Contour vert du masque ────────────────────────────────────────────────
    cnts, _ = cv2.findContours(inspection_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0, 230, 0), 2)

    # ── Contours blancs fins autour de chaque void ────────────────────────────
    if void_mask.any():
        vc, _ = cv2.findContours(void_mask.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, vc, -1, (255, 255, 255), 1)

    # ── Cadre + croix : plus gros void intérieur ──────────────────────────────
    if analysis_results.get("largest_void_bbox") is not None:
        mr, mc, xr, xc = analysis_results["largest_void_bbox"]
        cv2.rectangle(result, (mc-3, mr-3), (xc+3, xr+3), (80, 220, 255), 3)
        if analysis_results.get("largest_void_centroid"):
            cy, cx = map(int, analysis_results["largest_void_centroid"])
            cv2.line(result, (cx-16, cy), (cx+16, cy), (80, 220, 255), 2)
            cv2.line(result, (cx, cy-16), (cx, cy+16), (80, 220, 255), 2)

    return result


# ─── Compat ───────────────────────────────────────────────────────────────────

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
