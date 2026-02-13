import sys
import subprocess
import os
import io
import tempfile

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ─── Auto-installation de TensorFlow ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _install_tensorflow():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        with st.spinner("⏳ Installation de TensorFlow (une seule fois, ~2 min)..."):
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "tensorflow-cpu==2.13.0",
                "--quiet", "--no-warn-script-location"
            ])
        st.cache_resource.clear()
        st.rerun()

_install_tensorflow()

import tensorflow as tf
from tensorflow import keras

from utils.void_analysis_utils import (
    preprocess_image, apply_mask, analyze_voids,
    create_visualization, resize_with_aspect_ratio
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse RX – Détection de Voids",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1.5rem}
.alert-box{padding:.9rem;border-radius:.5rem;margin:.4rem 0}
.ok  {background:#d4edda;color:#155724;border:1px solid #c3e6cb}
.warn{background:#fff3cd;color:#856404;border:1px solid #ffeeba}
.bad {background:#f8d7da;color:#721c24;border:1px solid #f5c6cb}
.info{background:#e7f3ff;border:1px solid #b3d9ff}
.slider-group{background:#f8f9fa;padding:.7rem;border-radius:.4rem;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)

# ─── Helpers modèle ───────────────────────────────────────────────────────────

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    inter    = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.*inter + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

@st.cache_resource(show_spinner=False)
def load_model_from_path(tmp_path: str):
    return keras.models.load_model(tmp_path, compile=False)


# ─── Gestion du masque PNG ────────────────────────────────────────────────────

def decode_mask_png(uploaded_png) -> np.ndarray:
    """
    Lit le PNG uploadé et retourne un masque binaire (H×W) uint8 :
      255  = zone à inspecter (pixels verts dans le PNG)
      0    = zone exclue      (reste)
    """
    pil = Image.open(uploaded_png).convert("RGB")
    arr = np.array(pil)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green = (g > 150) & (r < 100) & (b < 100)
    mask  = (green.astype(np.uint8)) * 255
    return mask   # shape (H_mask, W_mask)


def transform_mask(mask_bin: np.ndarray,
                   target_h: int, target_w: int,
                   tx_pct: float, ty_pct: float,
                   scale: float, angle_deg: float) -> np.ndarray:
    """
    Applique échelle + rotation + translation au masque binaire,
    puis le place sur un canvas (target_h × target_w).

    tx_pct, ty_pct : décalage en % de la taille de l'image cible
                     (0 = pas de décalage, positif = vers droite/bas)
    scale          : facteur d'échelle (1.0 = taille d'origine)
    angle_deg      : rotation en degrés (sens horaire)

    Retourne un tableau (target_h, target_w, 3) uint8
    avec canal vert = zone à inspecter.
    """
    Hm, Wm = mask_bin.shape

    # 1. Redimensionner selon l'échelle
    new_w = max(1, int(Wm * scale))
    new_h = max(1, int(Hm * scale))
    scaled = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 2. Rotation autour du centre du masque
    cx, cy = new_w / 2, new_h / 2
    M_rot  = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    rotated = cv2.warpAffine(scaled, M_rot, (new_w, new_h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. Construire le canvas final (taille image RX)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    # Translation : décalage en pixels à partir du coin supérieur gauche
    tx_px = int(target_w * tx_pct / 100)
    ty_px = int(target_h * ty_pct / 100)

    # Position du masque sur le canvas (centré par défaut si tx=ty=0)
    x0 = (target_w - new_w) // 2 + tx_px
    y0 = (target_h - new_h) // 2 + ty_px

    # Intersection entre le canvas et le masque (clip aux bords)
    cx0 = max(0, -x0);    cy0 = max(0, -y0)
    cx1 = min(new_w, target_w - x0)
    cy1 = min(new_h, target_h - y0)
    dx0 = max(0, x0);     dy0 = max(0, y0)
    dx1 = dx0 + (cx1 - cx0)
    dy1 = dy0 + (cy1 - cy0)

    if cx1 > cx0 and cy1 > cy0:
        canvas[dy0:dy1, dx0:dx1] = rotated[cy0:cy1, cx0:cx1]

    # 4. Convertir en image couleur (vert = inspecté)
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    out[:,:,1] = canvas   # canal vert
    return out


def overlay_preview(image_rgb: np.ndarray, mask_color: np.ndarray) -> np.ndarray:
    """Superpose le masque sur l'image : zones exclues assombries, zone verte teintée."""
    green  = mask_color[:,:,1] > 100
    result = image_rgb.copy().astype(np.float32)
    result[~green] *= 0.4
    result[green, 1] = np.clip(result[green, 1] * 0.8 + 50, 0, 255)
    result = result.astype(np.uint8)
    contours, _ = cv2.findContours(green.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 220, 0), 3)
    return result


# ─── Traitement principal ─────────────────────────────────────────────────────

def process_image(image, mask, model, contrast, brightness, filter_geo):
    gray      = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape)==3 else image.copy()
    processed = preprocess_image(gray, contrast, brightness)
    masked_image, inspection_mask = apply_mask(processed, mask)

    resized, _ = resize_with_aspect_ratio(masked_image, (512, 512))
    inp = np.expand_dims(resized.astype(np.float32)/255.0, axis=(0,-1))
    pred = model.predict(inp, verbose=0)[0]
    pred = cv2.resize(pred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    results   = analyze_voids(pred, inspection_mask, filter_geo)
    vis_image = create_visualization(image, pred, inspection_mask, results)
    return vis_image, results


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🧠 Modèle")
        up_model = st.file_uploader("Fichier modèle (.h5)", type=["h5"])
        if up_model and st.button("🔄 Initialiser", use_container_width=True):
            with st.spinner("Chargement…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                    tmp.write(up_model.getvalue()); tmp_path = tmp.name
                model = load_model_from_path(tmp_path)
                os.remove(tmp_path)
                if model:
                    st.session_state["model"] = model
                    st.success("✅ Modèle chargé!")
        if "model" in st.session_state:
            st.caption("✅ Modèle actif")

        st.divider()
        st.subheader("🎛️ Prétraitement")
        contrast   = st.slider("Contraste",  0.5, 2.0, 1.0, 0.05)
        brightness = st.slider("Luminosité", -50,  50,   0,    5)

        st.divider()
        st.subheader("🔍 Analyse")
        filter_geo = st.checkbox("Filtrer formes géométriques", value=True)

    return contrast, brightness, filter_geo


# ─── PANNEAU MASQUE ───────────────────────────────────────────────────────────

def mask_panel(image_rgb: np.ndarray):
    """
    Upload du PNG masque + 4 sliders d'ajustement.
    Retourne le masque couleur (H×W×3) prêt pour l'analyse, ou None.
    """
    H, W = image_rgb.shape[:2]

    st.subheader("2️⃣ Masque d'inspection")

    # ── Upload ────────────────────────────────────────────────────────────────
    col_up, col_legend = st.columns([2, 1])
    with col_up:
        up_mask = st.file_uploader(
            "Charger le masque PNG",
            type=["png"],
            help="Vert (0,255,0) = zone inspectée · Noir = zone exclue"
        )
    with col_legend:
        st.markdown("""
        **Format du masque :**
        - 🟩 **Vert** → inspecté
        - ⬛ **Noir** → exclu
        """)

    if up_mask is None:
        st.info("Chargez un masque PNG pour continuer.")
        return None

    # Décoder le masque une seule fois (cache par nom de fichier)
    cache_key = f"mask_raw_{up_mask.name}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = decode_mask_png(up_mask)
    mask_raw = st.session_state[cache_key]

    if mask_raw.max() == 0:
        st.error("❌ Aucun pixel vert détecté. Vérifiez que le masque utilise R=0, G=255, B=0.")
        return None

    # ── Sliders d'ajustement ──────────────────────────────────────────────────
    st.markdown("**🔧 Ajustement du masque**")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("<div class='slider-group'>", unsafe_allow_html=True)
        st.caption("↔️ Décalage X")
        tx = st.slider("X (%)", -50, 50, 0, 1, key="tx",
                       help="Décale le masque horizontalement.\n0 = centré")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='slider-group'>", unsafe_allow_html=True)
        st.caption("↕️ Décalage Y")
        ty = st.slider("Y (%)", -50, 50, 0, 1, key="ty",
                       help="Décale le masque verticalement.\n0 = centré")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown("<div class='slider-group'>", unsafe_allow_html=True)
        st.caption("🔄 Rotation")
        angle = st.slider("Angle (°)", -180, 180, 0, 1, key="angle",
                          help="Fait pivoter le masque")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown("<div class='slider-group'>", unsafe_allow_html=True)
        st.caption("🔍 Échelle")
        scale = st.slider("Échelle", 0.1, 3.0, 1.0, 0.01, key="scale",
                          help="1.0 = taille d'origine du PNG")
        st.markdown("</div>", unsafe_allow_html=True)

    # Bouton reset
    c_reset, c_info = st.columns([1, 3])
    with c_reset:
        if st.button("↺ Réinitialiser", use_container_width=True):
            for k in ["tx","ty","angle","scale"]:
                st.session_state[k] = 0 if k != "scale" else 1.0
            st.rerun()
    with c_info:
        pct_src = mask_raw.mean() / 255 * 100
        st.caption(f"Masque source : {mask_raw.shape[1]}×{mask_raw.shape[0]} px — "
                   f"{pct_src:.1f}% de zone verte")

    # ── Transformer et prévisualiser ──────────────────────────────────────────
    mask_color = transform_mask(mask_raw, H, W, tx, ty, scale, angle)

    pct_final = (mask_color[:,:,1] > 100).mean() * 100
    if pct_final < 0.5:
        st.warning("⚠️ Le masque est entièrement hors de l'image — ajustez X/Y ou l'échelle.")

    col_prev1, col_prev2 = st.columns(2)
    with col_prev1:
        preview = overlay_preview(image_rgb, mask_color)
        st.image(preview,
                 caption=f"Prévisualisation — {pct_final:.1f}% de la surface inspectée",
                 use_container_width=True)
    with col_prev2:
        # Afficher le masque transformé seul (vert sur noir)
        mask_disp = np.zeros((H, W, 3), dtype=np.uint8)
        mask_disp[:,:,1] = mask_color[:,:,1]   # vert uniquement
        st.image(mask_disp, caption="Masque transformé seul", use_container_width=True)

    return mask_color


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    st.markdown('<h1 class="main-title">🔬 Analyse RX – Détection de Voids</h1>',
                unsafe_allow_html=True)

    contrast, brightness, filter_geo = sidebar()

    tab_analyse, tab_help = st.tabs(["📤 Analyse", "ℹ️ Instructions"])

    with tab_analyse:

        if "model" not in st.session_state:
            st.info("⬅️ Chargez d'abord un modèle dans la barre latérale.")
            st.stop()

        model = st.session_state["model"]

        # 1. Image RX
        st.subheader("1️⃣ Charger l'image RX")
        up_img = st.file_uploader("Image RX (.png / .jpg / .jpeg)",
                                  type=["png","jpg","jpeg"])
        if up_img is None:
            st.stop()

        raw      = np.frombuffer(up_img.read(), np.uint8)
        img_bgr  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Masque
        mask = mask_panel(image_rgb)
        if mask is None:
            st.stop()

        # 3. Analyse
        st.subheader("3️⃣ Lancer l'analyse")
        if st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyse en cours…"):
                vis_image, results = process_image(
                    image_rgb, mask, model, contrast, brightness, filter_geo
                )
            st.session_state["results"]   = results
            st.session_state["vis_image"] = vis_image

        # 4. Résultats
        if "results" in st.session_state:
            results   = st.session_state["results"]
            vis_image = st.session_state["vis_image"]

            st.success("✅ Analyse terminée!")
            st.subheader("4️⃣ Résultats")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Image originale**")
                st.image(image_rgb, use_container_width=True)
            with c2:
                st.markdown("**Image analysée**")
                st.caption("🔵 Bleu foncé = soudure · 🔴 Rouge = void · 🟦 Cadre = plus gros void")
                st.image(vis_image, use_container_width=True)

            # Métriques
            st.subheader("📊 Statistiques")
            vr = results["void_ratio"]
            lr = results["largest_void_ratio"]
            nv = results["num_voids"]

            def badge(v, t1, t2):
                return ("ok","✅") if v<t1 else (("warn","⚠️") if v<t2 else ("bad","❌"))

            cg, ig = badge(vr, 5, 15)
            cl, il = badge(lr, 2,  5)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="alert-box {cg}"><b>{ig} Taux global</b><br>'
                            f'<span style="font-size:1.8rem;font-weight:700">{vr:.2f} %</span></div>',
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="alert-box {cl}"><b>{il} Plus gros void</b><br>'
                            f'<span style="font-size:1.8rem;font-weight:700">{lr:.2f} %</span></div>',
                            unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="alert-box info"><b>📍 Nombre de voids</b><br>'
                            f'<span style="font-size:1.8rem;font-weight:700">{nv}</span></div>',
                            unsafe_allow_html=True)

            with st.expander("📋 Détails complets"):
                st.json({
                    "Surface inspectée (px)":     int(results["total_inspection_area"]),
                    "Surface soudure (px)":        int(results["soudure_area"]),
                    "Surface voids (px)":          int(results["voids_area"]),
                    "Taux voids (%)":              round(vr, 2),
                    "Surface plus gros void (px)": int(results["largest_void_area"]),
                    "Ratio plus gros void (%)":    round(lr, 2),
                    "Nombre de voids":             nv,
                })

            st.subheader("💾 Télécharger")
            d1, d2 = st.columns(2)
            with d1:
                buf = io.BytesIO()
                Image.fromarray(vis_image).save(buf, format="PNG")
                st.download_button("📥 Image analysée (PNG)", buf.getvalue(),
                                   "analyse_voids.png", "image/png",
                                   use_container_width=True)
            with d2:
                report = {"taux_manque_global_%": round(vr,2),
                          "taux_plus_gros_void_%": round(lr,2),
                          "nombre_voids": nv,
                          "surface_inspection_px": int(results["total_inspection_area"]),
                          "surface_soudure_px": int(results["soudure_area"]),
                          "surface_voids_px": int(results["voids_area"])}
                st.download_button("📥 Rapport JSON", str(report),
                                   "rapport_analyse.json", "application/json",
                                   use_container_width=True)

    with tab_help:
        st.markdown("""
## 📖 Guide d'utilisation

### 1. Charger le modèle *(barre latérale)*
1. Cliquez **"Fichier modèle (.h5)"** → sélectionnez `void_detection_best.h5`
2. Cliquez **"🔄 Initialiser"**

### 2. Charger l'image RX
Formats : PNG, JPG, JPEG.

### 3. Charger et ajuster le masque

**Format du masque PNG :**

| Couleur | Code | Rôle |
|---------|------|------|
| 🟩 Vert | `(0, 255, 0)` | Zone à inspecter |
| ⬛ Noir | `(0, 0, 0)` | Zone exclue |

**Créer un masque avec GIMP :**
1. Ouvrir l'image RX dans GIMP
2. Créer une nouvelle image vide de même taille, fond noir
3. Peindre en vert `(0, 255, 0)` la zone de soudure
4. Exporter en PNG → `masque_qfn.png`

**Ajustement avec les sliders :**

| Slider | Rôle |
|--------|------|
| **X (%)** | Décale horizontalement (-50 = gauche, +50 = droite) |
| **Y (%)** | Décale verticalement (-50 = haut, +50 = bas) |
| **Angle (°)** | Fait pivoter le masque autour de son centre |
| **Échelle** | 1.0 = taille originale, 0.5 = moitié, 2.0 = double |

La **prévisualisation** se met à jour en temps réel.
Deux vues : image avec masque superposé + masque seul (vert sur noir).

Le masque ajusté peut être réutilisé pour tous les composants du même type.

### 4. Prétraitement *(barre latérale)*
- **Contraste** : augmentez si les voids sont peu visibles (1.2–1.5)
- **Luminosité** : ajustez selon l'exposition du cliché

### 5. Résultats

| Couleur | Signification |
|---------|--------------|
| 🔵 Bleu foncé | Soudure |
| 🔴 Rouge | Void / manque |
| 🟦 Cadre bleu ciel | Plus gros void |

| Métrique | ✅ Bon | ⚠️ Acceptable | ❌ Non conforme |
|----------|--------|--------------|----------------|
| Taux global | < 5 % | 5–15 % | > 15 % |
| Plus gros void | < 2 % | 2–5 % | > 5 % |
        """)


if __name__ == "__main__":
    main()
