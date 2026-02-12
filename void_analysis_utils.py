"""
Fonctions utilitaires pour l'analyse d'images et la détection de voids
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology
from typing import Tuple, Dict, List


def preprocess_image(image: np.ndarray, adjust_contrast: float = 1.0, 
                     adjust_brightness: int = 0) -> np.ndarray:
    """
    Prétraite l'image avec ajustement de contraste et luminosité
    
    Args:
        image: Image en niveaux de gris
        adjust_contrast: Facteur de contraste (1.0 = pas de changement)
        adjust_brightness: Ajustement de luminosité (-100 à 100)
    
    Returns:
        Image prétraitée
    """
    # Ajustement du contraste et de la luminosité
    adjusted = cv2.convertScaleAbs(image, alpha=adjust_contrast, beta=adjust_brightness)
    
    # Réduction du bruit avec filtre bilatéral
    # Préserve les bords tout en lissant le bruit
    denoised = cv2.bilateralFilter(adjusted, 9, 75, 75)
    
    return denoised


def apply_mask(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique le masque d'inspection à l'image
    
    Args:
        image: Image d'entrée
        mask: Masque (vert = zone à inspecter, noir = zone d'exclusion)
    
    Returns:
        image_masked: Image avec masque appliqué
        inspection_mask: Masque binaire (1 = inspecter, 0 = exclure)
    """
    # Extraire la zone verte du masque
    if len(mask.shape) == 3:
        green_channel = mask[:, :, 1]
        red_channel = mask[:, :, 2]
        blue_channel = mask[:, :, 0]
        
        # Zone verte: G élevé, R et B faibles
        inspection_mask = (green_channel > 100) & (red_channel < 100) & (blue_channel < 100)
    else:
        # Si le masque est en niveaux de gris
        inspection_mask = mask > 127
    
    inspection_mask = inspection_mask.astype(np.uint8)
    
    # Appliquer le masque à l'image
    if len(image.shape) == 3:
        image_masked = image.copy()
        image_masked[inspection_mask == 0] = 0
    else:
        image_masked = cv2.bitwise_and(image, image, mask=inspection_mask)
    
    return image_masked, inspection_mask


def filter_geometric_shapes(binary_mask: np.ndarray, 
                           min_circularity: float = 0.6,
                           min_aspect_ratio: float = 0.3) -> np.ndarray:
    """
    Filtre les formes géométriques parfaites (pistes, vias, etc.)
    
    Args:
        binary_mask: Masque binaire des détections
        min_circularity: Circularité minimale pour considérer comme void (0-1)
        min_aspect_ratio: Ratio d'aspect minimal pour éviter les formes allongées
    
    Returns:
        Masque filtré sans les formes géométriques artificielles
    """
    # Labelliser les composants connectés
    labeled = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labeled)
    
    filtered_mask = np.zeros_like(binary_mask)
    
    for region in regions:
        # Calculer la circularité: 4π*area / perimeter²
        if region.perimeter == 0:
            continue
        
        circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
        
        # Calculer le ratio d'aspect
        if region.major_axis_length == 0:
            continue
        aspect_ratio = region.minor_axis_length / region.major_axis_length
        
        # Calculer l'écart par rapport à un rectangle parfait
        bbox_area = region.bbox_area
        extent = region.area / bbox_area if bbox_area > 0 else 0
        
        # Filtrer les formes trop parfaites (probablement des éléments du PCB)
        is_perfect_circle = circularity > 0.95 and aspect_ratio > 0.95
        is_perfect_rectangle = extent > 0.95 and aspect_ratio < 0.4
        is_elongated = aspect_ratio < min_aspect_ratio
        
        # Garder seulement les formes organiques (voids naturels)
        if not (is_perfect_circle or is_perfect_rectangle or is_elongated):
            coords = region.coords
            filtered_mask[coords[:, 0], coords[:, 1]] = 1
    
    return filtered_mask


def analyze_voids(prediction: np.ndarray, 
                  inspection_mask: np.ndarray,
                  filter_shapes: bool = True) -> Dict:
    """
    Analyse les voids détectés dans la prédiction
    
    Args:
        prediction: Prédiction du modèle (H, W, 3) avec softmax
        inspection_mask: Masque de la zone d'inspection
        filter_shapes: Si True, filtre les formes géométriques parfaites
    
    Returns:
        Dictionnaire avec les résultats d'analyse
    """
    # Extraire les classes
    soudure_mask = prediction[:, :, 0] > 0.5  # Classe 0: soudure
    voids_mask = prediction[:, :, 1] > 0.5     # Classe 1: voids/manques
    
    # Appliquer le masque d'inspection
    soudure_mask = soudure_mask & (inspection_mask > 0)
    voids_mask = voids_mask & (inspection_mask > 0)
    
    # Filtrer les formes géométriques si demandé
    if filter_shapes:
        voids_mask = filter_geometric_shapes(voids_mask.astype(np.uint8))
    
    # Calculer les surfaces
    total_inspection_area = np.sum(inspection_mask > 0)
    soudure_area = np.sum(soudure_mask)
    voids_area = np.sum(voids_mask)
    
    # Calculer le ratio de voids
    void_ratio = (voids_area / total_inspection_area * 100) if total_inspection_area > 0 else 0
    
    # Trouver le plus gros void
    labeled_voids = measure.label(voids_mask, connectivity=2)
    void_regions = measure.regionprops(labeled_voids)
    
    largest_void_area = 0
    largest_void_ratio = 0
    largest_void_bbox = None
    largest_void_centroid = None
    
    if len(void_regions) > 0:
        # Filtrer les voids qui touchent les bords du masque d'inspection
        interior_voids = []
        
        for region in void_regions:
            # Vérifier si le void touche les bords du masque
            minr, minc, maxr, maxc = region.bbox
            
            # Créer un masque légèrement élargi pour vérifier les bords
            border_width = 2
            touches_border = False
            
            # Vérifier les 4 côtés
            if minr < border_width or minc < border_width:
                touches_border = True
            if maxr > (inspection_mask.shape[0] - border_width):
                touches_border = True
            if maxc > (inspection_mask.shape[1] - border_width):
                touches_border = True
            
            # Vérifier si le void touche la frontière du masque d'inspection
            for coord in region.coords:
                y, x = coord
                # Vérifier dans un voisinage 3x3 s'il y a une zone exclue
                if y > 0 and y < inspection_mask.shape[0]-1 and x > 0 and x < inspection_mask.shape[1]-1:
                    neighborhood = inspection_mask[y-1:y+2, x-1:x+2]
                    if np.any(neighborhood == 0):
                        touches_border = True
                        break
            
            if not touches_border:
                interior_voids.append(region)
        
        # Trouver le plus gros void intérieur
        if len(interior_voids) > 0:
            largest_void = max(interior_voids, key=lambda r: r.area)
            largest_void_area = largest_void.area
            largest_void_ratio = (largest_void_area / total_inspection_area * 100) if total_inspection_area > 0 else 0
            largest_void_bbox = largest_void.bbox
            largest_void_centroid = largest_void.centroid
    
    # Créer les masques de visualisation
    soudure_viz = soudure_mask.astype(np.uint8) * 255
    voids_viz = voids_mask.astype(np.uint8) * 255
    
    results = {
        'void_ratio': void_ratio,
        'largest_void_ratio': largest_void_ratio,
        'largest_void_area': largest_void_area,
        'largest_void_bbox': largest_void_bbox,
        'largest_void_centroid': largest_void_centroid,
        'total_inspection_area': total_inspection_area,
        'soudure_area': soudure_area,
        'voids_area': voids_area,
        'num_voids': len(void_regions),
        'soudure_mask': soudure_viz,
        'voids_mask': voids_viz,
        'void_regions': void_regions
    }
    
    return results


def create_visualization(original_image: np.ndarray,
                        prediction: np.ndarray,
                        inspection_mask: np.ndarray,
                        analysis_results: Dict) -> np.ndarray:
    """
    Crée une visualisation colorée des résultats
    
    Args:
        original_image: Image originale
        prediction: Prédiction du modèle
        inspection_mask: Masque d'inspection
        analysis_results: Résultats de l'analyse
    
    Returns:
        Image RGB avec visualisation
    """
    # Créer une image RGB
    if len(original_image.shape) == 2:
        vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        vis_image = original_image.copy()
    
    h, w = vis_image.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Extraire les masques
    soudure_mask = prediction[:, :, 0] > 0.5
    voids_mask = prediction[:, :, 1] > 0.5
    
    # Appliquer le masque d'inspection
    soudure_mask = soudure_mask & (inspection_mask > 0)
    voids_mask = voids_mask & (inspection_mask > 0)
    
    # Colorer la soudure en bleu foncé (B=255, G=0, R=0)
    overlay[soudure_mask, 0] = 255  # Blue channel
    overlay[soudure_mask, 1] = 0    # Green channel
    overlay[soudure_mask, 2] = 0    # Red channel
    
    # Colorer les voids en rouge (B=0, G=0, R=255)
    overlay[voids_mask, 0] = 0      # Blue channel
    overlay[voids_mask, 1] = 0      # Green channel
    overlay[voids_mask, 2] = 255    # Red channel
    
    # Mélanger l'overlay avec l'image originale
    alpha = 0.5
    vis_image = cv2.addWeighted(vis_image, 1-alpha, overlay, alpha, 0)
    
    # Dessiner le contour du plus gros void en bleu ciel épais
    if analysis_results['largest_void_bbox'] is not None:
        bbox = analysis_results['largest_void_bbox']
        minr, minc, maxr, maxc = bbox
        
        # Bleu ciel: BGR = (255, 255, 135)
        thickness = 5
        cv2.rectangle(vis_image, (minc, minr), (maxc, maxr), (255, 255, 135), thickness)
        
        # Ajouter un cercle au centre
        if analysis_results['largest_void_centroid'] is not None:
            cy, cx = analysis_results['largest_void_centroid']
            cv2.circle(vis_image, (int(cx), int(cy)), 10, (255, 255, 135), thickness)
    
    return vis_image


def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int],
                             pad_color: int = 0) -> Tuple[np.ndarray, Tuple]:
    """
    Redimensionne l'image en conservant le ratio d'aspect
    
    Args:
        image: Image à redimensionner
        target_size: Taille cible (height, width)
        pad_color: Couleur de padding
    
    Returns:
        image_resized: Image redimensionnée avec padding
        (scale, pad_top, pad_left): Paramètres de transformation
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculer le ratio de redimensionnement
    scale = min(target_w / w, target_h / h)
    
    # Nouvelles dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionner
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculer le padding
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    # Ajouter le padding
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[pad_color, pad_color, pad_color]
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_color
        )
    
    return padded, (scale, pad_top, pad_left)


def inverse_resize(bbox: Tuple, transform_params: Tuple, 
                   original_size: Tuple[int, int]) -> Tuple:
    """
    Inverse la transformation de redimensionnement pour les coordonnées
    
    Args:
        bbox: Bounding box (minr, minc, maxr, maxc)
        transform_params: Paramètres de transformation (scale, pad_top, pad_left)
        original_size: Taille originale (height, width)
    
    Returns:
        Bounding box dans les coordonnées originales
    """
    scale, pad_top, pad_left = transform_params
    minr, minc, maxr, maxc = bbox
    
    # Retirer le padding
    minr = minr - pad_top
    minc = minc - pad_left
    maxr = maxr - pad_top
    maxc = maxc - pad_left
    
    # Inverser le scaling
    minr = int(minr / scale)
    minc = int(minc / scale)
    maxr = int(maxr / scale)
    maxc = int(maxc / scale)
    
    # Clamp aux dimensions originales
    h, w = original_size
    minr = max(0, min(minr, h))
    minc = max(0, min(minc, w))
    maxr = max(0, min(maxr, h))
    maxc = max(0, min(maxc, w))
    
    return (minr, minc, maxr, maxc)
