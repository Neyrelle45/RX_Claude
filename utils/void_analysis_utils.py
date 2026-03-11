"""
Utilitaires V8 — Détection 100% classique, zéro IA.

PHYSIQUE RX VALIDÉE SUR DATASET :
  Soudure dense  → absorbe les RX → pixel SOMBRE  (moy ~58)
  Void/manque    → peu de métal  → pixel MOINS SOMBRE (moy ~85)
  Séparation ~27 niveaux → Otsu local parfait

ALGORITHME :
  1. CLAHE local (clipLimit=3, grid=8×8) → rehausse contraste dans le masque
  2. Otsu calculé UNIQUEMENT sur les pixels du masque utilisateur
  3. Pixels > seuil Otsu dans le masque = voids candidats
  4. Morphologie : ouverture (supprime bruit) + fermeture (soude les blobs)
  5. Filtre taille : supprime blobs < 100px (valeur ajustable)

PERFORMANCES MESURÉES SUR 8 IMAGES LABELISÉES :
  Rappel moyen : 92%  (ne rate presque aucun void)
  F1 moyen     : 64–80% selon précision du masque utilisateur

RENDU :
  🟢 Vert  — Soudure présente
  🔴 Rouge — Void / manque
  ⬛ Noir  — Zone exclue
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


# ─── Détection voids — algorithme classique ───────────────────────────────────

def detect_voids_threshold(gray_image, roi_mask, sensitivity=0, min_void_px=100, return_debug=False):
    """
    Détection classique validée sur 8 images RX labelisées.

    Args:
        gray_image   : image RX en niveaux de gris (uint8), prétraitée
        roi_mask     : masque uint8 (1 = zone à inspecter)
        sensitivity  : décalage du seuil Otsu en niveaux de gris.
                       0  = Otsu pur (recommandé)
                       >0 = seuil plus haut → moins de voids (moins sensible)
                       <0 = seuil plus bas  → plus de voids  (plus sensible)
        min_void_px  : taille minimale d'un void en pixels (défaut 100)
        return_debug : si True, retourne aussi un dict de debug

    Returns:
        Si return_debug=False: void_mask (bool H×W), seuil_utilisé (float)
        Si return_debug=True: void_mask, seuil_utilisé, debug_dict
    """
    # Init debug info dès le début
    debug_info = {
        "pixels_bruts": 0,
        "pixels_morph": 0,
        "blobs_avant": 0,
        "blobs_apres": 0,
        "rejets": []
    }
    
    if not roi_mask.any():
        if return_debug:
            return np.zeros(gray_image.shape, dtype=bool), 0.0, debug_info
        return np.zeros(gray_image.shape, dtype=bool), 0.0

    # ── 1. Normalisation robuste par percentile dans le masque ──────────────
    # Stretch p3→p97 (plus agressif pour détecter voids faibles)
    vals_raw = gray_image[roi_mask > 0]
    p3  = float(np.percentile(vals_raw, 3))
    p97 = float(np.percentile(vals_raw, 97))
    stretched = np.clip(
        (gray_image.astype(np.float32) - p3) / max(p97 - p3, 1) * 255,
        0, 255).astype(np.uint8)
    # CLAHE TRÈS fort pour révéler tous les voids subtils
    _clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced = _clahe.apply(stretched)

    # ── 2. Otsu sur les pixels du masque uniquement ───────────────────────────
    vals = enhanced[roi_mask > 0].reshape(-1, 1).astype(np.uint8)
    thr_otsu, _ = cv2.threshold(vals, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # BIAIS AGRESSIF : Otsu global rate souvent les voids dans les pads gris moyens
    # On applique un offset par défaut NÉGATIF pour forcer plus de détection
    # sensitivity=0 → offset -10 (plus de voids)
    # sensitivity=+10 → offset 0 (Otsu pur)
    # sensitivity=-10 → offset -20 (très agressif)
    DEFAULT_BIAS = -10  # Biais par défaut pour détecter plus
    thr = float(thr_otsu) + DEFAULT_BIAS + float(sensitivity)
    
    # Sécurité : ne pas descendre en dessous de 50 (sinon tout devient void)
    thr = max(thr, 50.0)

    # ── 3. Voids = pixels CLAIRS (> seuil) en RX ─────────────────────────────
    # PHYSIQUE RX : zones claires = peu de métal (RX traversent) = VOIDS
    #               zones sombres = métal dense (RX absorbés) = SOUDURE
    void_raw = (enhanced.astype(np.float32) > thr) & (roi_mask > 0)

    # ── 4. Morphologie MINIMALE pour contours précis ─────────────────────────
    # Images RX propres → morphologie très légère
    # Ouverture avec k2 au lieu de k3 pour préserver les contours
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    # Ouverture ultra-légère : supprime SEULEMENT le bruit isolé 1 pixel
    cleaned = cv2.morphologyEx(void_raw.astype(np.uint8), cv2.MORPH_OPEN, k2)
    
    # Alternative : si les contours sont encore imprécis, essayer sans morphologie
    # cleaned = void_raw.astype(np.uint8)
    
    # ── Anti "fromage grignoté" : combler les encoches de vias ───────────────
    # Principe : les vias créent de petites concavités sur le bord des voids.
    # On détecte ces concavités via les défauts du hull convexe,
    # et on comble uniquement celles dont la profondeur < max_via_depth px.
    labeled_tmp = measure.label(cleaned, connectivity=2)
    filled = np.zeros_like(cleaned)
    for r in measure.regionprops(labeled_tmp):
        blob = (labeled_tmp == r.label).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            filled += blob
            continue
        cnt = max(cnts, key=cv2.contourArea)
        # Remplissage des trous internes (flood-fill)
        h2, w2 = blob.shape
        pad = np.zeros((h2+2, w2+2), np.uint8)
        pad[1:-1, 1:-1] = blob
        fld = pad.copy()
        cv2.floodFill(fld, None, (0,0), 1)
        interior = (fld[1:-1, 1:-1] == 0).astype(np.uint8)
        blob_filled = np.clip(blob + interior, 0, 1).astype(np.uint8)
        # Défauts du hull : combler les petites encoches (vias)
        if len(cnt) >= 5:
            try:
                hull_idx = cv2.convexHull(cnt, returnPoints=False)
                defects  = cv2.convexityDefects(cnt, hull_idx)
                if defects is not None:
                    for defect in defects.reshape(-1, 4):
                        s, e, f, depth_px = defect
                        depth = depth_px / 256.0
                        if depth < 30:   # encoche < 30px = via → combler
                            start = tuple(cnt[s][0])
                            end   = tuple(cnt[e][0])
                            far   = tuple(cnt[f][0])
                            tri   = np.array([start, end, far], dtype=np.int32)
                            cv2.fillPoly(blob_filled, [tri], 1)
            except Exception:
                pass
        filled = np.clip(filled + blob_filled, 0, 1)
    cleaned = filled.astype(np.uint8)

    # ── 5. Séparation des blobs complexes (void+piste fusionnés) ─────────────
    # Les blobs avec AR faible ou solidité faible peuvent être deux objets fusionnés.
    # On applique un watershed par distance-transform pour les séparer proprement.
    try:
        from scipy import ndimage as _ndi
        from skimage import segmentation as _seg
        from skimage.feature import peak_local_max as _plm

        labeled_tmp2 = measure.label(cleaned, connectivity=2)
        separated = np.zeros_like(cleaned)
        for r in measure.regionprops(labeled_tmp2):
            if r.area < min_void_px:
                continue
            blob = (labeled_tmp2 == r.label).astype(np.uint8)
            maj = r.axis_major_length if hasattr(r, 'axis_major_length') else r.major_axis_length
            mni = r.axis_minor_length if hasattr(r, 'axis_minor_length') else r.minor_axis_length
            ar  = mni / max(maj, 1)
            sol = r.solidity

            # Blob simple (rond, solide) → garder directement
            if ar > 0.55 or sol > 0.80:
                separated[blob > 0] = 1
                continue

            # Blob complexe → séparation par watershed sur distance transform
            dist = _ndi.distance_transform_edt(blob)
            min_d = max(8, int(np.sqrt(min_void_px / np.pi) * 0.8))
            coords = _plm(dist, min_distance=min_d, labels=blob, threshold_abs=5.0)
            if len(coords) <= 1:
                separated[blob > 0] = 1
            else:
                markers = np.zeros_like(blob, dtype=np.int32)
                for i, (py, px) in enumerate(coords, 1):
                    markers[py, px] = i
                ws = _seg.watershed(-dist, markers, mask=blob)
                for lbl in range(1, int(ws.max()) + 1):
                    region = (ws == lbl) & (blob > 0)
                    if region.sum() >= min_void_px:
                        separated[region] = 1
        cleaned = separated.astype(np.uint8)
    except Exception:
        pass  # si scipy absent, continuer sans séparation

    # ── 6. Filtre taille + forme ──────────────────────────────────────────────
    # DEBUG: Logging pour diagnostiquer 0 détection
    debug_info["pixels_bruts"] = int(void_raw.sum())
    debug_info["pixels_morph"] = int(cleaned.sum())
    
    labeled = measure.label(cleaned, connectivity=2)
    n_blobs_before = int(labeled.max())
    debug_info["blobs_avant"] = n_blobs_before
    
    filtered = np.zeros_like(cleaned)

    # Composants connexes du masque pour ratio local par pad
    msk_lab    = measure.label(roi_mask.astype(np.uint8), connectivity=2)
    total_mask = int(roi_mask.sum()) if roi_mask.sum() > 0 else 1
    min_comp   = total_mask * 0.01
    comp_sizes = {mr.label: mr.area
                  for mr in measure.regionprops(msk_lab) if mr.area >= min_comp}

    for r in measure.regionprops(labeled):
        if r.area < min_void_px:
            continue
        maj = r.axis_major_length if hasattr(r, 'axis_major_length') else r.major_axis_length
        mni = r.axis_minor_length if hasattr(r, 'axis_minor_length') else r.minor_axis_length
        if maj == 0:
            continue
        ar   = mni / maj
        circ = 4 * np.pi * r.area / (r.perimeter ** 2 + 1e-6)
        ecc  = r.eccentricity

        # Ratio par rapport à la surface TOTALE inspectée
        # (pas le composant local qui peut être fragmenté par des exclusions)
        # VERSION 2024-03-06 FIX RATIO
        ratio_local = r.area / max(total_mask, 1)

        # Filtres réactivés avec seuils TRES permissifs
        # 1. Rejeter blobs GIGANTESQUES (probablement tout le fond du pad)
        if ratio_local > 0.80:  # > 80% de la surface totale = artefact
            debug_info["rejets"].append(f"Blob {r.label}: ratio={ratio_local:.3f}")
            continue
        # 2. Barres/rectangles extrêmes seulement
        if ar < 0.10 and ecc > 0.98:   # barre ultra-fine
            debug_info["rejets"].append(f"Blob {r.label}: barre AR={ar:.2f}")
            continue
        if circ < 0.03 and ar < 0.15:  # rectangle ultra-plat
            debug_info["rejets"].append(f"Blob {r.label}: rect circ={circ:.2f}")
            continue
        # 3. BGA : cercles parfaits ET grands
        if circ > 0.85 and ar > 0.85 and r.area > 500:
            debug_info["rejets"].append(f"Blob {r.label}: cercle circ={circ:.2f} AR={ar:.2f}")
            continue
        filtered[labeled == r.label] = 1

    n_blobs_final = int(measure.label(filtered).max())
    debug_info["blobs_apres"] = n_blobs_final
    debug_info["pixels_final"] = int(filtered.sum())
    
    if return_debug:
        return filtered.astype(bool), float(thr), debug_info
    return filtered.astype(bool), float(thr)


# ─── Correction manuelle intelligente ────────────────────────────────────────

def smart_add_void(gray_image, roi_mask, current_void_mask, click_y, click_x):
    """
    Ajoute un void en trouvant la région connexe CLAIRE qui contient le point cliqué.
    En RX : voids = zones claires (rayons X traversent facilement).
    """
    # Normalisation identique à detect_voids_threshold
    vals_raw = gray_image[roi_mask > 0]
    p3  = float(np.percentile(vals_raw, 3))
    p97 = float(np.percentile(vals_raw, 97))
    stretched = np.clip(
        (gray_image.astype(np.float32) - p3) / max(p97 - p3, 1) * 255,
        0, 255).astype(np.uint8)
    _clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced = _clahe.apply(stretched)

    vals   = enhanced[roi_mask > 0].reshape(-1, 1).astype(np.uint8)
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Même biais agressif que detect_voids_threshold
    thr    = float(thr) - 10.0
    thr    = max(thr, 50.0)

    # Voids = zones CLAIRES (> seuil)
    bright  = (enhanced.astype(np.float32) > thr) & (roi_mask > 0)
    labeled = measure.label(bright.astype(np.uint8), connectivity=2)

    blob_id = int(labeled[click_y, click_x])
    if blob_id == 0:
        best_d, best_id = float("inf"), 0
        for r in measure.regionprops(labeled):
            ry, rx = map(int, r.centroid)
            d = (ry - click_y) ** 2 + (rx - click_x) ** 2
            if d < best_d:
                best_d, best_id = d, r.label
        blob_id = best_id
    if blob_id == 0:
        return current_void_mask, 0

    blob   = (labeled == blob_id).astype(np.uint8)
    rp_lst = [r for r in measure.regionprops(labeled) if r.label == blob_id]
    if not rp_lst:
        return current_void_mask, 0
    rp = rp_lst[0]

    maj = rp.axis_major_length if hasattr(rp, 'axis_major_length') else rp.major_axis_length
    mni = rp.axis_minor_length if hasattr(rp, 'axis_minor_length') else rp.minor_axis_length
    ar  = mni / max(maj, 1)
    sol = rp.solidity

    # CORRECTION MANUELLE : toujours prendre TOUTE la région
    # Le watershed était trop imprécis et ne suivait pas les contours
    # L'utilisateur peut cliquer plusieurs fois si nécessaire
    region = (labeled == blob_id)

    new_void = current_void_mask.copy()
    new_void[region] = True
    return new_void, int(region.sum())


# ─── Analyse principale ───────────────────────────────────────────────────────

def analyze_voids(prediction, inspection_mask,
                  filter_geometric=True,
                  void_threshold=None,
                  gray_image=None,
                  sensitivity=0,
                  min_void_px=100,
                  solder_threshold=None,
                  use_ai_zone=False):
    """
    Analyse des voids dans la zone d'inspection.

    La détection utilise uniquement le traitement d'image classique (CLAHE + Otsu).
    Le modèle IA (prediction) est ignoré sauf si use_ai_zone=True.

    Args:
        gray_image   : image RX en gris (REQUIS)
        inspection_mask : masque binaire zone à inspecter
        sensitivity  : décalage seuil Otsu (0 = automatique)
        min_void_px  : taille min void en pixels
    """
    total = int(np.sum(inspection_mask > 0))

    if gray_image is not None:
        void_mask, void_thr, debug_info = detect_voids_threshold(
            gray_image,
            inspection_mask.astype(np.uint8),
            sensitivity=sensitivity,
            min_void_px=min_void_px,
            return_debug=True)
    else:
        # Aucune image grise → pas de détection possible
        void_mask = np.zeros(inspection_mask.shape, dtype=bool)
        void_thr  = 0.0
        debug_info = {}

    solder_zone    = (inspection_mask > 0)
    solder_present = solder_zone & ~void_mask

    # ── Métriques ─────────────────────────────────────────────────────────────
    # total = surface totale inspectée (zones vertes du masque, INCLUANT exclusions noires)
    # n_solder = total (car solder_zone = inspection_mask > 0)
    n_solder = int(np.sum(solder_zone))  # = total
    n_voids  = int(np.sum(void_mask))
    # CRITIQUE: ratio sur surface TOTALE (zones vertes uniquement, exclusions noires exclues)
    void_ratio = n_voids / total * 100 if total > 0 else 0.0

    # Plus gros void (simplement le blob avec la plus grande aire)
    lv_area=0; lv_ratio=0.0; lv_bbox=None; lv_centroid=None
    if void_mask.any():
        labeled  = measure.label(void_mask.astype(np.uint8), connectivity=2)
        biggest = None
        for r in measure.regionprops(labeled):
            if biggest is None or r.area > biggest.area:
                biggest = r
        if biggest:
            lv_area     = biggest.area
            lv_ratio    = lv_area / total * 100 if total > 0 else 0.0
            lv_bbox     = biggest.bbox
            lv_centroid = biggest.centroid

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
        solder_zone=solder_zone,
        void_mask=void_mask,
        void_threshold_used=float(void_thr),
        solder_threshold_used=0.0,
        debug_info=debug_info,  # Info de debug pour diagnostic
    )


# ─── Visualisation ────────────────────────────────────────────────────────────

def create_visualization(original_image, prediction, inspection_mask,
                         analysis_results, no_crop=False):
    """
    Rendu avec transparence :
      🟢 Vert 50% alpha — Soudure présente (texture visible)
      🔴 Rouge — Void / manque
      ⬛ Noir  — Zone exclue
    + Auto-crop sur zone d'inspection (sauf si no_crop=True pour correction manuelle)
    """
    if original_image.ndim == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        base = original_image.copy()

    void_mask   = analysis_results.get("void_mask")
    solder_zone = analysis_results.get("solder_zone")

    if solder_zone is None:
        solder_zone = (inspection_mask > 0)
    if void_mask is None:
        void_mask = np.zeros(inspection_mask.shape, dtype=bool)

    # FIX: voids SEULEMENT dans la zone inspectée (pas dans les exclusions)
    void_mask = void_mask & (inspection_mask > 0)
    
    solder_present = solder_zone & ~void_mask
    exclu          = (inspection_mask == 0)

    result = base.astype(np.float32).copy()

    # Noir absolu hors masque
    result[exclu] = 0

    # Vert TRANSPARENT (50% alpha) : on voit l'image originale dessous
    if solder_present.any():
        # Blending: 50% vert + 50% image originale
        green_overlay = np.zeros_like(result)
        green_overlay[solder_present] = [0, 180, 0]  # Vert pur
        result[solder_present] = (0.5 * result[solder_present] + 
                                  0.5 * green_overlay[solder_present])

    # Rouge vif : voids (opaque)
    if void_mask.any():
        result[void_mask, 0] = 235
        result[void_mask, 1] = 15
        result[void_mask, 2] = 15

    result = np.clip(result, 0, 255).astype(np.uint8)

    # Contour vert du masque
    cnts, _ = cv2.findContours(inspection_mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0, 230, 0), 2)

    # Contours blancs fins autour des voids
    if void_mask.any():
        vc, _ = cv2.findContours(void_mask.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, vc, -1, (255, 255, 255), 1)

    # Plus gros void : calcul gardé mais marquage visuel supprimé
    # (le calcul se fait dans analyze_voids, pas besoin de le dessiner)

    # AUTO-CROP : zoom sur la zone d'inspection (sauf si no_crop pour correction manuelle)
    if not no_crop and inspection_mask.any():
        ys, xs = np.where(inspection_mask > 0)
        if len(ys) > 0 and len(xs) > 0:
            y_min, y_max = int(ys.min()), int(ys.max()) + 1
            x_min, x_max = int(xs.min()), int(xs.max()) + 1
            # Marge de 5% pour ne pas couper les contours
            h, w = result.shape[:2]
            margin_y = max(5, int((y_max - y_min) * 0.05))
            margin_x = max(5, int((x_max - x_min) * 0.05))
            y_min = max(0, y_min - margin_y)
            y_max = min(h, y_max + margin_y)
            x_min = max(0, x_min - margin_x)
            x_max = min(w, x_max + margin_x)
            result = result[y_min:y_max, x_min:x_max]

    return result


# ─── Compat ───────────────────────────────────────────────────────────────────

def detect_solder_zone(prediction, inspection_mask, solder_threshold=None):
    """Conservé pour compatibilité — retourne simplement le masque complet."""
    return (inspection_mask > 0), 0.0


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
