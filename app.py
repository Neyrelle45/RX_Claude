import sys
import subprocess
import os
import io
import tempfile
import json
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd

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
.archive-row{border-bottom:1px solid #eee;padding:.4rem 0}
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

def get_model_input_size(model) -> tuple:
    """Lit la taille d'entrée directement depuis le modèle pour éviter tout mismatch."""
    try:
        shape = model.input_shape  # ex: (None, 384, 384, 1)
        return int(shape[1]), int(shape[2])   # (H, W)
    except Exception:
        return 512, 512   # fallback

# ─── Masque PNG ───────────────────────────────────────────────────────────────
def decode_mask_png(uploaded_png) -> np.ndarray:
    """PNG -> masque binaire (H×W) uint8 : 255 = vert, 0 = noir."""
    pil = Image.open(uploaded_png).convert("RGB")
    arr = np.array(pil)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green = (g > 150) & (r < 100) & (b < 100)
    return (green.astype(np.uint8)) * 255

def transform_mask(mask_bin: np.ndarray,
                   target_h: int, target_w: int,
                   tx_pct: float, ty_pct: float,
                   scale: float, angle_deg: float) -> np.ndarray:
    """Echelle + rotation + translation → masque couleur (H×W×3)."""
    Hm, Wm = mask_bin.shape
    new_w = max(1, int(Wm * scale))
    new_h = max(1, int(Hm * scale))
    scaled = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    cx, cy = new_w / 2, new_h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    rotated = cv2.warpAffine(scaled, M, (new_w, new_h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    tx_px  = int(target_w * tx_pct / 100)
    ty_px  = int(target_h * ty_pct / 100)
    x0 = (target_w - new_w) // 2 + tx_px
    y0 = (target_h - new_h) // 2 + ty_px
    cx0 = max(0, -x0);   cy0 = max(0, -y0)
    cx1 = min(new_w, target_w - x0)
    cy1 = min(new_h, target_h - y0)
    dx0 = max(0, x0);    dy0 = max(0, y0)
    dx1 = dx0 + (cx1 - cx0)
    dy1 = dy0 + (cy1 - cy0)
    if cx1 > cx0 and cy1 > cy0:
        canvas[dy0:dy1, dx0:dx1] = rotated[cy0:cy1, cx0:cx1]

    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    out[:,:,1] = canvas
    return out

def overlay_preview(image_rgb, mask_color):
    green  = mask_color[:,:,1] > 100
    result = image_rgb.copy().astype(np.float32)
    result[~green] *= 0.4
    result[green, 1] = np.clip(result[green, 1] * 0.8 + 50, 0, 255)
    result = result.astype(np.uint8)
    contours, _ = cv2.findContours(green.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 220, 0), 3)
    return result

# ─── Traitement image ─────────────────────────────────────────────────────────
def process_image(image_rgb, mask_color, model, contrast, brightness, filter_geo):
    """
    Pipeline complet. Détecte automatiquement la taille d'entrée du modèle.
    """
    # 1. Niveaux de gris
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Prétraitement
    processed = preprocess_image(gray, contrast, brightness)

    # 3. Extraire masque binaire (H×W) et aligner sur l'image
    H_img, W_img = processed.shape[:2]
    if mask_color.ndim == 3:
        green_ch = mask_color[:,:,1]
        red_ch   = mask_color[:,:,2]
        blue_ch  = mask_color[:,:,0]
        bin_mask = ((green_ch > 100) & (red_ch < 100) & (blue_ch < 100)).astype(np.uint8)
    else:
        bin_mask = (mask_color > 127).astype(np.uint8)

    if bin_mask.shape != (H_img, W_img):
        bin_mask = cv2.resize(bin_mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask > 0).astype(np.uint8)

    # 4. Appliquer le masque → image niveaux de gris avec zones exclues = 0
    masked = cv2.bitwise_and(processed, processed, mask=bin_mask)

    # 5. Taille d'entrée réelle du modèle (lue depuis ses métadonnées)
    TARGET_H, TARGET_W = get_model_input_size(model)

    # 6. Redimensionner avec conservation du ratio + padding
    resized, _ = resize_with_aspect_ratio(masked, (TARGET_H, TARGET_W))

    # Garantir la taille exacte après padding éventuel
    if resized.shape[0] != TARGET_H or resized.shape[1] != TARGET_W:
        resized = cv2.resize(resized, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

    # 7. Construire tenseur (1, H, W, 1) sans ambiguïté
    arr = resized.astype(np.float32) / 255.0
    if arr.ndim == 2:
        inp = arr[np.newaxis, :, :, np.newaxis]          # (1, H, W, 1) ✓
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
        inp = arr[np.newaxis, :, :, np.newaxis]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        inp = arr[np.newaxis, :, :, :]
    else:
        raise ValueError(f"Shape inattendue: {arr.shape}")

    # 8. Prédiction
    pred = model.predict(inp, verbose=0)[0]              # (H_model, W_model, 3)

    # 9. Remettre à la taille de l'image originale
    pred_full = cv2.resize(pred, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    # 10. Analyse et visualisation
    results   = analyze_voids(pred_full, bin_mask, filter_geo)
    vis_image = create_visualization(image_rgb, pred_full, bin_mask, results)
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
                    h, w = get_model_input_size(model)
                    st.success(f"✅ Modèle chargé — entrée {h}×{w}")
        if "model" in st.session_state:
            h, w = get_model_input_size(st.session_state["model"])
            st.caption(f"✅ Modèle actif · entrée {h}×{w} px")

        st.divider()
        st.subheader("🎛️ Prétraitement")
        contrast   = st.slider("Contraste",  0.5, 2.0, 1.0, 0.05)
        brightness = st.slider("Luminosité", -50,  50,   0,    5)

        st.divider()
        st.subheader("🔍 Analyse")
        filter_geo = st.checkbox("Filtrer formes géométriques", value=True)

    return contrast, brightness, filter_geo

# ─── PANNEAU MASQUE ───────────────────────────────────────────────────────────
def mask_panel(image_rgb):
    H, W = image_rgb.shape[:2]
    st.subheader("2️⃣ Masque d'inspection")

    col_up, col_leg = st.columns([2, 1])
    with col_up:
        up_mask = st.file_uploader("Charger le masque PNG",
                                   type=["png"],
                                   help="Vert (0,255,0)=inspecté · Noir=exclu")
    with col_leg:
        st.markdown("**Format PNG :**\n- 🟩 Vert → inspecté\n- ⬛ Noir → exclu")

    if up_mask is None:
        st.info("Chargez un masque PNG pour continuer.")
        return None

    cache_key = f"mask_raw_{up_mask.name}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = decode_mask_png(up_mask)
    mask_raw = st.session_state[cache_key]

    if mask_raw.max() == 0:
        st.error("❌ Aucun pixel vert détecté — vérifiez la couleur (R=0, G=255, B=0).")
        return None

    st.markdown("**🔧 Ajustement du masque**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.caption("↔️ X")
        tx = st.slider("X (%)", -50, 50, 0, 1, key="tx")
    with c2:
        st.caption("↕️ Y")
        ty = st.slider("Y (%)", -50, 50, 0, 1, key="ty")
    with c3:
        st.caption("🔄 Rotation")
        angle = st.slider("Angle (°)", -180, 180, 0, 1, key="angle")
    with c4:
        st.caption("🔍 Échelle")
        scale = st.slider("Échelle", 0.1, 3.0, 1.0, 0.01, key="scale")

    cr, ci = st.columns([1, 3])
    with cr:
        if st.button("↺ Réinitialiser", use_container_width=True):
            for k in ["tx","ty","angle"]: st.session_state[k] = 0
            st.session_state["scale"] = 1.0
            st.rerun()
    with ci:
        pct_src = mask_raw.mean() / 255 * 100
        st.caption(f"Source : {mask_raw.shape[1]}×{mask_raw.shape[0]} px · {pct_src:.1f}% vert")

    mask_color = transform_mask(mask_raw, H, W, tx, ty, scale, angle)
    pct = (mask_color[:,:,1] > 100).mean() * 100
    if pct < 0.5:
        st.warning("⚠️ Masque hors image — ajustez X/Y ou l'échelle.")

    cp1, cp2 = st.columns(2)
    with cp1:
        st.image(overlay_preview(image_rgb, mask_color),
                 caption=f"Prévisualisation — {pct:.1f}% inspecté",
                 use_container_width=True)
    with cp2:
        disp = np.zeros((H, W, 3), dtype=np.uint8)
        disp[:,:,1] = mask_color[:,:,1]
        st.image(disp, caption="Masque seul", use_container_width=True)

    return mask_color

# ─── ARCHIVE ──────────────────────────────────────────────────────────────────
def init_archive():
    if "archive" not in st.session_state:
        st.session_state["archive"] = []

def archive_result(filename, results, vis_image):
    """Ajoute une entrée à l'archive en session."""
    buf = io.BytesIO()
    Image.fromarray(vis_image).save(buf, format="PNG")
    entry = {
        "ts":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fichier":   filename,
        "taux_%":    round(results["void_ratio"], 2),
        "plus_gros_%": round(results["largest_void_ratio"], 2),
        "nb_voids":  results["num_voids"],
        "img_bytes": buf.getvalue(),
    }
    st.session_state["archive"].append(entry)

def show_archive():
    """Affiche le tableau d'archive avec vignettes cliquables."""
    st.subheader("🗄️ Archive des résultats")
    archive = st.session_state.get("archive", [])

    if not archive:
        st.info("Aucun résultat archivé. Cliquez sur **'📥 Archiver ce résultat'** après une analyse.")
        return

    # Bouton vider
    col_dl, col_vide = st.columns([2, 1])
    with col_vide:
        if st.button("🗑️ Vider tous les résultats", use_container_width=True, type="secondary"):
            st.session_state["archive"] = []
            st.rerun()

    # Export CSV
    with col_dl:
        rows = [{k: v for k, v in e.items() if k != "img_bytes"} for e in archive]
        csv  = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter CSV", csv, "archive_voids.csv", "text/csv",
                           use_container_width=True)

    st.divider()

    # Tableau + vignettes
    for i, entry in enumerate(archive):
        def badge_color(v, t1, t2):
            return "🟢" if v < t1 else ("🟡" if v < t2 else "🔴")

        bg = badge_color(entry["taux_%"], 5, 15)
        bl = badge_color(entry["plus_gros_%"], 2, 5)

        with st.container():
            col_img, col_data, col_dl2 = st.columns([1, 3, 1])

            with col_img:
                # Vignette cliquable via expander
                with st.expander("🔍 Voir", expanded=False):
                    st.image(entry["img_bytes"], use_container_width=True)

            with col_data:
                st.markdown(
                    f"**{entry['fichier']}** &nbsp;·&nbsp; `{entry['ts']}`\n\n"
                    f"| Taux global | Plus gros void | Nb voids |\n"
                    f"|:-----------:|:--------------:|:--------:|\n"
                    f"| {bg} **{entry['taux_%']}%** | {bl} **{entry['plus_gros_%']}%** "
                    f"| {entry['nb_voids']} |"
                )

            with col_dl2:
                st.download_button(
                    "📥 PNG",
                    data=entry["img_bytes"],
                    file_name=f"analyse_{entry['fichier']}",
                    mime="image/png",
                    use_container_width=True,
                    key=f"dl_{i}"
                )
            st.divider()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    init_archive()
    st.markdown('<h1 class="main-title">🔬 Analyse RX – Détection de Voids</h1>',
                unsafe_allow_html=True)

    contrast, brightness, filter_geo = sidebar()

    tab_analyse, tab_archive, tab_help = st.tabs(["📤 Analyse", "🗄️ Archive", "ℹ️ Instructions"])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_analyse:

        if "model" not in st.session_state:
            st.info("⬅️ Chargez d'abord un modèle dans la barre latérale.")
            st.stop()
        model = st.session_state["model"]

        st.subheader("1️⃣ Charger l'image RX")
        up_img = st.file_uploader("Image RX (.png / .jpg / .jpeg)",
                                  type=["png","jpg","jpeg"])
        if up_img is None:
            st.stop()

        raw      = np.frombuffer(up_img.read(), np.uint8)
        img_bgr  = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = mask_panel(image_rgb)
        if mask is None:
            st.stop()

        st.subheader("3️⃣ Lancer l'analyse")
        if st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyse en cours…"):
                vis_image, results = process_image(
                    image_rgb, mask, model, contrast, brightness, filter_geo
                )
            st.session_state["results"]    = results
            st.session_state["vis_image"]  = vis_image
            st.session_state["last_fname"] = up_img.name

        if "results" in st.session_state:
            results   = st.session_state["results"]
            vis_image = st.session_state["vis_image"]
            fname     = st.session_state.get("last_fname", "image.png")

            st.success("✅ Analyse terminée!")

            # ── Images côte à côte ────────────────────────────────────────────
            st.subheader("4️⃣ Résultats")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Image originale**")
                st.image(image_rgb, use_container_width=True)
            with c2:
                st.markdown("**Image analysée**")
                st.caption("🔵 Soudure · 🔴 Void/manque · 🟦 Cadre = plus gros void")
                st.image(vis_image, use_container_width=True)

            # ── Tableau de résultats ──────────────────────────────────────────
            st.subheader("📊 Résultats de l'analyse")

            vr = results["void_ratio"]
            lr = results["largest_void_ratio"]
            nv = results["num_voids"]

            def status(v, t1, t2):
                if v < t1: return "✅ Bon"
                if v < t2: return "⚠️ Acceptable"
                return "❌ Non conforme"

            df = pd.DataFrame([{
                "Métrique":              "Taux de manque global",
                "Valeur":               f"{vr:.2f} %",
                "Seuil conforme":       "< 5 %",
                "Seuil acceptable":     "< 15 %",
                "Statut":               status(vr, 5, 15),
            },{
                "Métrique":              "Plus gros void (intérieur)",
                "Valeur":               f"{lr:.2f} %",
                "Seuil conforme":       "< 2 %",
                "Seuil acceptable":     "< 5 %",
                "Statut":               status(lr, 2, 5),
            },{
                "Métrique":              "Nombre de voids détectés",
                "Valeur":               str(nv),
                "Seuil conforme":       "—",
                "Seuil acceptable":     "—",
                "Statut":               "ℹ️",
            },{
                "Métrique":              "Surface inspectée",
                "Valeur":               f"{results['total_inspection_area']:,} px",
                "Seuil conforme":       "—",
                "Seuil acceptable":     "—",
                "Statut":               "ℹ️",
            },{
                "Métrique":              "Surface voids",
                "Valeur":               f"{results['voids_area']:,} px",
                "Seuil conforme":       "—",
                "Seuil acceptable":     "—",
                "Statut":               "ℹ️",
            }])

            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Métriques visuelles ───────────────────────────────────────────
            def badge(v, t1, t2):
                return ("ok","✅") if v<t1 else (("warn","⚠️") if v<t2 else ("bad","❌"))

            cg, ig = badge(vr, 5, 15)
            cl, il = badge(lr, 2, 5)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="alert-box {cg}"><b>{ig} Taux global</b><br>'
                            f'<span style="font-size:1.8rem;font-weight:700">{vr:.2f} %</span></div>',
                            unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="alert-box {cl}"><b>{il} Plus gros void</b><br>'
                            f'<span style="font-size:1.8rem;font-weight:700">{lr:.2f} %</span></div>',
                            unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="alert-box info"><b>📍 Nb voids</b><br>'
                            f'<span style="font-size:1.8rem;font-weight:700">{nv}</span></div>',
                            unsafe_allow_html=True)

            # ── Actions ───────────────────────────────────────────────────────
            st.subheader("💾 Actions")
            a1, a2, a3 = st.columns(3)

            with a1:
                buf = io.BytesIO()
                Image.fromarray(vis_image).save(buf, format="PNG")
                st.download_button("📥 Télécharger PNG", buf.getvalue(),
                                   f"analyse_{fname}", "image/png",
                                   use_container_width=True)
            with a2:
                report = {"fichier": fname, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "taux_%": round(vr,2), "plus_gros_%": round(lr,2), "nb_voids": nv}
                st.download_button("📥 Télécharger JSON", json.dumps(report, indent=2),
                                   f"rapport_{fname}.json", "application/json",
                                   use_container_width=True)
            with a3:
                if st.button("📥 Archiver ce résultat", use_container_width=True, type="secondary"):
                    archive_result(fname, results, vis_image)
                    st.success("✅ Résultat archivé ! Consultez l'onglet 🗄️ Archive.")

    # ══════════════════════════════════════════════════════════════════════════
    with tab_archive:
        show_archive()

    # ══════════════════════════════════════════════════════════════════════════
    with tab_help:
        st.markdown("""
## 📖 Guide d'utilisation

### 1. Charger le modèle *(barre latérale)*
1. **"Fichier modèle (.h5)"** → `void_detection_best.h5`
2. **"🔄 Initialiser"** — la taille d'entrée du modèle s'affiche automatiquement

### 2. Charger l'image RX
Formats : PNG, JPG, JPEG.

### 3. Charger et ajuster le masque
| Slider | Rôle |
|--------|------|
| **X (%)** | Décalage horizontal (-50 = gauche, +50 = droite) |
| **Y (%)** | Décalage vertical (-50 = haut, +50 = bas) |
| **Angle (°)** | Rotation autour du centre du masque |
| **Échelle** | 1.0 = taille PNG d'origine |

Format PNG : 🟩 **Vert** = inspecté · ⬛ **Noir** = exclu

### 4. Analyser → **🚀 Analyser**

### 5. Résultats

| Couleur image | Signification |
|---------------|--------------|
| 🔵 Bleu foncé | Soudure |
| 🔴 Rouge | Void / manque |
| 🟦 Cadre épais | Plus gros void identifié |

| Métrique | ✅ Bon | ⚠️ Acceptable | ❌ Non conforme |
|----------|--------|--------------|----------------|
| Taux global | < 5 % | 5–15 % | > 15 % |
| Plus gros void | < 2 % | 2–5 % | > 5 % |

### 6. Archive
- **📥 Archiver ce résultat** : stocke l'image analysée + métriques en session
- Onglet **🗄️ Archive** : tableau récapitulatif, vignettes cliquables, export CSV
- **🗑️ Vider tous les résultats** : efface l'archive de la session en cours
        """)

if __name__ == "__main__":
    main()
