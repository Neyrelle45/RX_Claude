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

import tensorflow as tf       # noqa: E402
from tensorflow import keras  # noqa: E402
# ──────────────────────────────────────────────────────────────────────────────

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
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    inter = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

@st.cache_resource(show_spinner=False)
def load_model_from_path(tmp_path: str):
    """Charge le modèle Keras depuis un chemin temporaire."""
    return keras.models.load_model(tmp_path, compile=False)


def build_mask(h: int, w: int,
               cx_pct: float, cy_pct: float,
               sw_pct: float, sh_pct: float,
               angle_deg: float) -> np.ndarray:
    """
    Crée un masque vert (zone à inspecter) en appliquant
    translation (cx, cy), échelle (sw, sh) et rotation.

    Paramètres exprimés en % de la taille de l'image.
    """
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Demi-dimensions du rectangle en pixels
    hw = int(w * sw_pct / 200)   # moitié largeur
    hh = int(h * sh_pct / 200)   # moitié hauteur

    # Centre en pixels
    cx = int(w * cx_pct / 100)
    cy = int(h * cy_pct / 100)

    # Coins du rectangle non-rotaté (relatifs au centre)
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float32)

    # Matrice de rotation
    rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    # Appliquer rotation puis translation
    rotated = (R @ corners.T).T + np.array([cx, cy])
    pts = rotated.astype(np.int32)

    # Remplir le polygone en vert
    cv2.fillPoly(mask, [pts], (0, 255, 0))
    return mask


def overlay_mask_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Superpose le contour vert du masque sur l'image pour prévisualisation."""
    preview = image.copy()
    green = mask[:, :, 1] > 0
    # Contour épais
    contours, _ = cv2.findContours(
        green.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(preview, contours, -1, (0, 230, 0), 3)
    # Remplissage semi-transparent
    overlay = preview.copy()
    overlay[green] = (overlay[green] * 0.6 + np.array([0, 200, 0]) * 0.4).astype(np.uint8)
    return overlay


def process_image(image, mask, model, contrast, brightness, filter_geometric):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    processed = preprocess_image(gray, contrast, brightness)
    masked_image, inspection_mask = apply_mask(processed, mask)

    input_size = (512, 512)
    resized, _ = resize_with_aspect_ratio(masked_image, input_size)
    model_input = resized.astype(np.float32) / 255.0
    model_input = np.expand_dims(model_input, axis=(0, -1))

    prediction = model.predict(model_input, verbose=0)[0]
    prediction_resized = cv2.resize(
        prediction, (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    results = analyze_voids(prediction_resized, inspection_mask, filter_geometric)
    vis_image = create_visualization(image, prediction_resized, inspection_mask, results)
    return vis_image, results


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")

        # ── Modèle ──────────────────────────────────────────────────────────
        st.subheader("🧠 Modèle")
        uploaded_model = st.file_uploader(
            "Charger le fichier .h5",
            type=["h5"],
            help="Fichier void_detection_best.h5 issu de l'entraînement"
        )
        if uploaded_model and st.button("🔄 Initialiser le modèle", use_container_width=True):
            with st.spinner("Chargement…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                    tmp.write(uploaded_model.getvalue())
                    tmp_path = tmp.name
                model = load_model_from_path(tmp_path)
                os.remove(tmp_path)
                if model is not None:
                    st.session_state["model"] = model
                    st.success("✅ Modèle chargé!")

        st.divider()

        # ── Prétraitement ───────────────────────────────────────────────────
        st.subheader("🎛️ Prétraitement")
        contrast   = st.slider("Contraste",   0.5, 2.0, 1.0, 0.05)
        brightness = st.slider("Luminosité",  -50,  50,   0,    5)

        st.divider()

        # ── Analyse ─────────────────────────────────────────────────────────
        st.subheader("🔍 Analyse")
        filter_geo = st.checkbox(
            "Filtrer formes géométriques",
            value=True,
            help="Exclut vias et pistes (cercles/rectangles parfaits)"
        )

    return contrast, brightness, filter_geo


# ─── MASK PANEL ───────────────────────────────────────────────────────────────
def mask_panel(image_rgb: np.ndarray):
    """
    Retourne le masque construit à partir des sliders,
    et affiche la prévisualisation.
    """
    h, w = image_rgb.shape[:2]

    st.subheader("2️⃣ Positionner le masque d'inspection")
    st.caption(
        "Ajustez position, taille et rotation du rectangle d'inspection. "
        "La zone **verte** sera analysée."
    )

    col_sliders, col_preview = st.columns([1, 2])

    with col_sliders:
        st.markdown("**📍 Position du centre (%)**")
        cx = st.slider("X  (gauche ↔ droite)", 0, 100, 50, 1, key="cx")
        cy = st.slider("Y  (haut ↔ bas)",       0, 100, 50, 1, key="cy")

        st.markdown("**📐 Dimensions (%)**")
        sw = st.slider("Largeur",  5, 100, 70, 1, key="sw")
        sh = st.slider("Hauteur",  5, 100, 70, 1, key="sh")

        st.markdown("**🔄 Rotation (°)**")
        angle = st.slider("Angle", -180, 180, 0, 1, key="angle")

        if st.button("↺ Réinitialiser le masque", use_container_width=True):
            for k in ["cx", "cy", "sw", "sh", "angle"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    # Construire le masque
    mask = build_mask(h, w, cx, cy, sw, sh, angle)

    with col_preview:
        preview = overlay_mask_preview(image_rgb, mask)
        st.image(preview, caption="Prévisualisation — zone verte inspectée",
                 use_container_width=True)

    return mask


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="main-title">🔬 Analyse RX – Détection de Voids</h1>',
                unsafe_allow_html=True)

    contrast, brightness, filter_geo = sidebar()

    tab_analyse, tab_help = st.tabs(["📤 Analyse", "ℹ️ Instructions"])

    # ── Onglet Analyse ────────────────────────────────────────────────────────
    with tab_analyse:

        if "model" not in st.session_state:
            st.info("⬅️ Chargez d'abord un modèle dans la barre latérale.")
            st.stop()

        model = st.session_state["model"]

        # 1. Upload image
        st.subheader("1️⃣ Charger l'image RX")
        uploaded_file = st.file_uploader(
            "Image RX (.png / .jpg / .jpeg)",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is None:
            st.stop()

        # Décoder l'image
        raw = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. Masque interactif
        mask = mask_panel(image_rgb)

        # 3. Analyse
        st.subheader("3️⃣ Lancer l'analyse")
        if st.button("🚀 Analyser", type="primary", use_container_width=True):
            if np.sum(mask[:, :, 1]) == 0:
                st.error("❌ Le masque est vide — ajustez les sliders.")
                st.stop()

            with st.spinner("🔄 Analyse en cours…"):
                vis_image, results = process_image(
                    image_rgb, mask, model,
                    contrast, brightness, filter_geo
                )
            st.session_state["results"]   = results
            st.session_state["vis_image"] = vis_image

        # 4. Résultats
        if "results" in st.session_state:
            results   = st.session_state["results"]
            vis_image = st.session_state["vis_image"]

            st.success("✅ Analyse terminée!")
            st.subheader("4️⃣ Résultats")

            # Images côte à côte
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Image originale**")
                st.image(image_rgb, use_container_width=True)
            with c2:
                st.markdown("**Image analysée**")
                st.caption("🔵 Bleu foncé = soudure · 🔴 Rouge = void/manque · 🟦 Bleu ciel = plus gros void")
                st.image(vis_image, use_container_width=True)

            # Métriques
            st.subheader("📊 Statistiques")
            vr = results["void_ratio"]
            lr = results["largest_void_ratio"]
            nv = results["num_voids"]

            def badge(val, t1, t2):
                if val < t1:   return "ok",   "✅"
                if val < t2:   return "warn", "⚠️"
                return "bad", "❌"

            cls_g, ico_g = badge(vr, 5, 15)
            cls_l, ico_l = badge(lr, 2,  5)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="alert-box {cls_g}">
                    <b>{ico_g} Taux de manque global</b><br>
                    <span style="font-size:1.8rem;font-weight:700">{vr:.2f} %</span>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="alert-box {cls_l}">
                    <b>{ico_l} Plus gros void</b><br>
                    <span style="font-size:1.8rem;font-weight:700">{lr:.2f} %</span>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="alert-box info">
                    <b>📍 Nombre de voids</b><br>
                    <span style="font-size:1.8rem;font-weight:700">{nv}</span>
                </div>""", unsafe_allow_html=True)

            with st.expander("📋 Détails complets"):
                st.json({
                    "Surface inspectée (px)":    int(results["total_inspection_area"]),
                    "Surface soudure (px)":       int(results["soudure_area"]),
                    "Surface voids (px)":         int(results["voids_area"]),
                    "Taux voids (%)":             round(vr, 2),
                    "Surface plus gros void (px)":int(results["largest_void_area"]),
                    "Ratio plus gros void (%)":   round(lr, 2),
                    "Nombre de voids":            nv,
                })

            # Téléchargements
            st.subheader("💾 Télécharger")
            d1, d2 = st.columns(2)
            with d1:
                buf = io.BytesIO()
                Image.fromarray(vis_image).save(buf, format="PNG")
                st.download_button(
                    "📥 Image analysée",
                    data=buf.getvalue(),
                    file_name="analyse_voids.png",
                    mime="image/png",
                    use_container_width=True
                )
            with d2:
                report = {
                    "taux_manque_global_%":  round(vr, 2),
                    "taux_plus_gros_void_%": round(lr, 2),
                    "nombre_voids":          nv,
                    "surface_inspection_px": int(results["total_inspection_area"]),
                    "surface_soudure_px":    int(results["soudure_area"]),
                    "surface_voids_px":      int(results["voids_area"]),
                }
                st.download_button(
                    "📥 Rapport JSON",
                    data=str(report),
                    file_name="rapport_analyse.json",
                    mime="application/json",
                    use_container_width=True
                )

    # ── Onglet Instructions ───────────────────────────────────────────────────
    with tab_help:
        st.markdown("""
## 📖 Guide d'utilisation

### 1. Charger le modèle *(barre latérale)*
1. Cliquez sur **"Charger le fichier .h5"**
2. Sélectionnez `void_detection_best.h5` (issu de l'entraînement Colab)
3. Cliquez **"Initialiser le modèle"**

### 2. Charger une image RX
Formats acceptés : PNG, JPG, JPEG.

### 3. Positionner le masque d'inspection
Utilisez les **5 sliders** pour définir la zone à analyser :

| Slider | Rôle |
|--------|------|
| **X** | Déplace le centre horizontalement (% largeur) |
| **Y** | Déplace le centre verticalement (% hauteur) |
| **Largeur** | Étire/rétrécit horizontalement |
| **Hauteur** | Étire/rétrécit verticalement |
| **Rotation** | Fait pivoter le rectangle (-180° → +180°) |

La **zone verte** sur la prévisualisation = zone qui sera analysée.

### 4. Régler le prétraitement *(barre latérale)*
- **Contraste** : augmentez si les voids sont peu visibles (1.2–1.5)
- **Luminosité** : ajustez selon l'exposition du cliché

### 5. Lancer l'analyse
Cliquez **🚀 Analyser**.

### 6. Interpréter les résultats

| Couleur | Signification |
|---------|--------------|
| 🔵 Bleu foncé | Soudure détectée |
| 🔴 Rouge | Void / manque de soudure |
| 🟦 Cadre bleu ciel épais | Plus gros void identifié |

#### Seuils indicatifs (IPC-7093 / J-STD-001)
| Métrique | ✅ Bon | ⚠️ Acceptable | ❌ Non conforme |
|----------|--------|--------------|----------------|
| Taux global | < 5 % | 5–15 % | > 15 % |
| Plus gros void | < 2 % | 2–5 % | > 5 % |

### 7. Export
- **PNG** : image annotée haute résolution
- **JSON** : données brutes pour traçabilité MES/ERP
        """)


if __name__ == "__main__":
    main()
