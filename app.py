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
    create_visualization, resize_with_aspect_ratio,
    remove_padding_and_restore
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
.legend-box{display:flex;flex-wrap:wrap;gap:1rem;padding:.8rem;
            background:#f8f9fa;border-radius:.4rem;margin:.5rem 0}
.legend-item{display:flex;align-items:center;gap:.4rem;font-size:.85rem}
.swatch{width:20px;height:20px;border-radius:3px;border:1px solid #ccc;flex-shrink:0}
</style>
""", unsafe_allow_html=True)

# ─── Constantes visuelles (synchronisées avec create_visualization) ───────────
COLORS = {
    "soudure": {"bgr": (255,  0,  0), "rgb": (0,   0, 255), "hex": "#0000ff",
                "label": "Soudure détectée"},
    "void":    {"bgr": (  0,  0,255), "rgb": (255, 0,   0), "hex": "#ff0000",
                "label": "Void / manque de soudure"},
    "cadre":   {"bgr": (255,255,135), "rgb": (135,255, 255), "hex": "#87ffff",
                "label": "Contour du plus gros void (intérieur)"},
    "exclu":   {"bgr": (  0,  0,  0), "rgb": (0,   0,   0), "hex": "#000000",
                "label": "Zone exclue du masque"},
}

# ─── Modèle ───────────────────────────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    inter    = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.*inter+smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)+smooth)

@st.cache_resource(show_spinner=False)
def load_model_from_path(tmp_path: str):
    return keras.models.load_model(tmp_path, compile=False)

def get_model_input_size(model) -> tuple:
    try:
        s = model.input_shape
        return int(s[1]), int(s[2])
    except Exception:
        return 384, 384

# ─── Prétraitement avancé ─────────────────────────────────────────────────────

def preprocess_advanced(gray: np.ndarray,
                        contrast: float, brightness: int,
                        clahe_clip: float, clahe_grid: int,
                        sharpen: float) -> np.ndarray:
    """
    Pipeline de prétraitement optimisé pour images RX de soudure.

    1. Contraste/luminosité linéaires  (base)
    2. CLAHE  – contraste adaptatif local, révèle les détails fins dans la soudure
    3. Filtre bilatéral  – réduit le bruit tout en préservant les contours
    4. Masque de netteté  – accentue les bords soudure/void
    """
    # 1. Contraste + luminosité linéaires
    img = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #    clahe_clip = limite d'amplification (2–8) ; élevé = plus de contraste local
    #    clahe_grid = taille de la grille (2–16) ; petit = plus local
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                                 tileGridSize=(clahe_grid, clahe_grid))
        img = clahe.apply(img)

    # 3. Débruitage bilatéral (préserve les contours)
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # 4. Masque de netteté (unsharp mask)
    if sharpen > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1 + sharpen, blurred, -sharpen, 0)
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img

# ─── Masque PNG ───────────────────────────────────────────────────────────────
def decode_mask_png(uploaded_png) -> np.ndarray:
    pil = Image.open(uploaded_png).convert("RGB")
    arr = np.array(pil)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    return ((g > 150) & (r < 100) & (b < 100)).astype(np.uint8) * 255

def transform_mask(mask_bin, target_h, target_w, tx_pct, ty_pct, scale, angle_deg):
    Hm, Wm = mask_bin.shape
    new_w = max(1, int(Wm * scale))
    new_h = max(1, int(Hm * scale))
    scaled = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    M = cv2.getRotationMatrix2D((new_w/2, new_h/2), -angle_deg, 1.0)
    rotated = cv2.warpAffine(scaled, M, (new_w, new_h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    tx_px  = int(target_w * tx_pct / 100)
    ty_px  = int(target_h * ty_pct / 100)
    x0 = (target_w - new_w) // 2 + tx_px
    y0 = (target_h - new_h) // 2 + ty_px
    cx0=max(0,-x0); cy0=max(0,-y0)
    cx1=min(new_w, target_w-x0); cy1=min(new_h, target_h-y0)
    dx0=max(0,x0);  dy0=max(0,y0)
    if cx1>cx0 and cy1>cy0:
        canvas[dy0:dy0+(cy1-cy0), dx0:dx0+(cx1-cx0)] = rotated[cy0:cy1, cx0:cx1]
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    out[:,:,1] = canvas
    return out

def overlay_preview(image_rgb, mask_color):
    green  = mask_color[:,:,1] > 100
    result = image_rgb.copy().astype(np.float32)
    result[~green] *= 0.4
    result[green, 1] = np.clip(result[green, 1]*0.8+50, 0, 255)
    result = result.astype(np.uint8)
    cnts, _ = cv2.findContours(green.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cnts, -1, (0,220,0), 3)
    return result

# ─── Process ──────────────────────────────────────────────────────────────────
def process_image(image_rgb, mask_color, model,
                  contrast, brightness, clahe_clip, clahe_grid, sharpen,
                  filter_geo, void_thr=None):
    H_img, W_img = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Masque binaire aligné sur l'image originale
    if mask_color.ndim == 3:
        bin_mask = ((mask_color[:,:,1]>100) & (mask_color[:,:,2]<100) &
                    (mask_color[:,:,0]<100)).astype(np.uint8)
    else:
        bin_mask = (mask_color>127).astype(np.uint8)
    if bin_mask.shape != (H_img, W_img):
        bin_mask = cv2.resize(bin_mask,(W_img,H_img),interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask>0).astype(np.uint8)

    # 2. Prétraitement
    processed = preprocess_advanced(gray, contrast, brightness,
                                    clahe_clip, clahe_grid, sharpen)
    masked = cv2.bitwise_and(processed, processed, mask=bin_mask)

    # 3. Resize vers taille modèle EN CONSERVANT LE RATIO (padding noir)
    TH, TW = get_model_input_size(model)
    # transform mémorise scale + padding pour inverser sans distorsion
    resized, transform = resize_with_aspect_ratio(masked, (TH, TW))
    # resize_with_aspect_ratio garantit déjà (TH,TW) — vérif de sécurité
    assert resized.shape[:2] == (TH, TW), f"Shape inattendue: {resized.shape}"

    # 4. Tenseur (1, TH, TW, 1)
    arr = resized.astype(np.float32) / 255.0
    inp = arr[np.newaxis, :, :, np.newaxis]   # toujours 2D gris ici

    # 5. Prédiction → (TH, TW, 3) avec padding
    pred_padded = model.predict(inp, verbose=0)[0]

    # 6. Enlever le padding PUIS resize vers la taille originale
    #    → aucune distorsion car on utilise exactement le même ratio
    pred_full = remove_padding_and_restore(pred_padded, transform)

    # 7. Analyse + visualisation
    results   = analyze_voids(pred_full, bin_mask, filter_geo, void_threshold=void_thr)
    vis_image = create_visualization(image_rgb, pred_full, bin_mask, results)
    return vis_image, results, pred_full, processed

# ─── Preview prétraitement live ───────────────────────────────────────────────
def preprocess_preview(image_rgb, contrast, brightness,
                       clahe_clip, clahe_grid, sharpen):
    """Retourne l'image prétraitée + statistiques pour preview live."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    proc = preprocess_advanced(gray, contrast, brightness,
                               clahe_clip, clahe_grid, sharpen)
    return cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def sidebar(image_rgb_ref):
    """
    Retourne les paramètres. Si image_rgb_ref fournie, affiche preview live.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Modèle
        st.subheader("🧠 Modèle")
        up_model = st.file_uploader("Fichier modèle (.h5)", type=["h5"])
        if up_model and st.button("🔄 Initialiser", use_container_width=True):
            with st.spinner("Chargement…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                    tmp.write(up_model.getvalue()); tmp_path=tmp.name
                model = load_model_from_path(tmp_path)
                os.remove(tmp_path)
                if model:
                    st.session_state["model"] = model
                    h,w = get_model_input_size(model)
                    st.success(f"✅ Modèle chargé — entrée {h}×{w}")
        if "model" in st.session_state:
            h,w = get_model_input_size(st.session_state["model"])
            st.caption(f"✅ Modèle actif · entrée {h}×{w} px")

        st.divider()

        # Prétraitement
        st.subheader("🎛️ Prétraitement")

        with st.expander("ℹ️ Guide des filtres", expanded=False):
            st.markdown("""
**Contraste** : amplifie les niveaux de gris globalement.
Utile si l'image est trop terne (valeur > 1.0).

**Luminosité** : décale tous les pixels vers le clair/sombre.

**CLAHE – Clip** : contraste adaptatif *local*.
⭐ C'est le paramètre le plus important pour les images RX.
Une valeur de 3–6 révèle les voids sombres dans la soudure brillante.
Trop élevé (>8) = bruit amplifié.

**CLAHE – Grille** : taille de la zone locale (px).
Petit (4–8) = très local. Grand (16+) = quasi-global.

**Netteté** : accentue les bords soudure/void.
Utile pour les images légèrement floues (0.3–0.8).
            """)

        contrast   = st.slider("Contraste",     0.5, 2.0, 1.0, 0.05, key="k_contrast")
        brightness = st.slider("Luminosité",    -50,  50,   0,    5,  key="k_brightness")
        clahe_clip = st.slider("CLAHE – Clip",  0.0, 10.0, 3.0, 0.5,  key="k_clahe_clip")
        clahe_grid = st.slider("CLAHE – Grille", 4,   32,   8,   2,   key="k_clahe_grid")
        sharpen    = st.slider("Netteté",       0.0,  2.0, 0.3, 0.1,  key="k_sharpen")

        # Preview live
        if image_rgb_ref is not None:
            st.caption("👁️ Preview prétraitement (live)")
            prev = preprocess_preview(image_rgb_ref, contrast, brightness,
                                      clahe_clip, clahe_grid, sharpen)
            st.image(prev, use_container_width=True)

        st.divider()
        st.subheader("🔍 Analyse")
        filter_geo = st.checkbox("Filtrer formes géométriques", value=True,
                                 help="Exclut vias et pistes (cercles/rectangles parfaits)")

        st.markdown("**Seuil de détection void**")
        auto_thr = st.checkbox("Seuil adaptatif (recommandé)", value=True,
                               help="Percentile 85 des probabilités dans la ROI — s'ajuste automatiquement")
        if auto_thr:
            void_thr = None
            st.caption("Le seuil sera calculé automatiquement à chaque analyse.")
        else:
            void_thr = st.slider("Seuil manuel", 0.01, 0.60, 0.10, 0.01,
                                 help="Plus bas = plus sensible. Avec un petit dataset, 0.03-0.15 est souvent optimal.")

    return contrast, brightness, clahe_clip, clahe_grid, sharpen, filter_geo, void_thr

# ─── MASQUE ───────────────────────────────────────────────────────────────────
def mask_panel(image_rgb):
    H, W = image_rgb.shape[:2]
    st.subheader("2️⃣ Masque d'inspection")

    col_up, col_leg = st.columns([2, 1])
    with col_up:
        up_mask = st.file_uploader("Charger le masque PNG", type=["png"],
                                   help="Vert (0,255,0)=inspecté · Noir=exclu")
    with col_leg:
        st.markdown("""
**Format PNG :**
- 🟩 **Vert** `(0,255,0)` → zone inspectée
- ⬛ **Noir** `(0,0,0)` → zone exclue
- ⚫ **Trous noirs** dans le vert → exclusions locales (ex: billes BGA)

*Le masque peut avoir des trous noirs pour exclure des zones précises
 à l'intérieur de la zone verte (ex: billes de soudure isolées).*
        """)

    if up_mask is None:
        st.info("Chargez un masque PNG pour continuer.")
        return None

    cache_key = f"mask_raw_{up_mask.name}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = decode_mask_png(up_mask)
    mask_raw = st.session_state[cache_key]

    if mask_raw.max() == 0:
        st.error("❌ Aucun pixel vert — vérifiez R=0, G=255, B=0.")
        return None

    st.markdown("**🔧 Ajustement du masque**")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.caption("↔️ X");      tx    = st.slider("X (%)",     -50,50,  0,1, key="tx")
    with c2: st.caption("↕️ Y");      ty    = st.slider("Y (%)",     -50,50,  0,1, key="ty")
    with c3: st.caption("🔄 Angle");  angle = st.slider("Angle (°)",-180,180, 0,1, key="angle")
    with c4: st.caption("🔍 Échelle");scale = st.slider("Échelle",   0.1,3.0,1.0,.01,key="scale")

    cr, ci = st.columns([1,3])
    with cr:
        if st.button("↺ Réinitialiser", use_container_width=True):
            for k in ["tx","ty","angle"]: st.session_state[k]=0
            st.session_state["scale"]=1.0
            st.rerun()
    with ci:
        pct_src = mask_raw.mean()/255*100
        st.caption(f"Source : {mask_raw.shape[1]}×{mask_raw.shape[0]} px · "
                   f"{pct_src:.1f}% vert")

    mask_color = transform_mask(mask_raw, H, W, tx, ty, scale, angle)
    pct = (mask_color[:,:,1]>100).mean()*100
    if pct < 0.5:
        st.warning("⚠️ Masque hors image — ajustez X/Y ou l'échelle.")

    cp1, cp2 = st.columns(2)
    with cp1:
        st.image(overlay_preview(image_rgb, mask_color),
                 caption=f"Prévisualisation — {pct:.1f}% de la surface inspectée",
                 use_container_width=True)
    with cp2:
        disp = np.zeros((H,W,3), dtype=np.uint8)
        disp[:,:,1] = mask_color[:,:,1]
        st.image(disp, caption="Masque seul (vert=inspecté, noir=exclu)",
                 use_container_width=True)

    return mask_color

# ─── LÉGENDE ──────────────────────────────────────────────────────────────────
def show_color_legend():
    st.markdown("""
<div class="legend-box">
  <div class="legend-item">
    <div class="swatch" style="background:#0000ff"></div>
    <span><b>Bleu foncé</b> — Soudure détectée (probabilité canal 0 &gt; 0.5)</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:#ff0000"></div>
    <span><b>Rouge</b> — Void / manque de soudure (probabilité canal 1 &gt; 0.5)</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:#87ffff;border:1px solid #aaa"></div>
    <span><b>Cadre bleu ciel épais</b> — Contour + centre du plus gros void intérieur
    (excluant les voids touchant le bord du masque)</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:#111;border:1px solid #888"></div>
    <span><b>Noir</b> — Zone exclue par le masque (non analysée)</span>
  </div>
</div>
""", unsafe_allow_html=True)

def show_heatmap_legend():
    st.markdown("""
<div class="legend-box">
  <div class="legend-item">
    <div class="swatch" style="background:linear-gradient(to right,#000,#fff)"></div>
    <span><b>Canal 0 — Soudure</b> : blanc = forte probabilité de soudure.
    Doit être brillant sur les zones de soudure.</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:linear-gradient(to right,#000,#ff4400)"></div>
    <span><b>Canal 1 — Voids/Manques</b> : rouge/jaune = forte probabilité de void.
    Doit s'allumer sur les zones sombres de la soudure.</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:linear-gradient(to right,#004040,#00ffcc)"></div>
    <span><b>Canal 2 — Fond</b> : cyan = zones hors composant.
    Doit être actif à l'extérieur de la soudure.</span>
  </div>
  <div class="legend-item" style="margin-top:.3rem">
    <span>📊 Les valeurs <i>min/max/moy</i> indiquent la plage de confiance du modèle.
    Un canal void avec moy&gt;0.1 signifie que des voids ont été détectés.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── ARCHIVE ──────────────────────────────────────────────────────────────────
def init_archive():
    if "archive" not in st.session_state:
        st.session_state["archive"] = []

def archive_result(filename, results, vis_image):
    buf = io.BytesIO()
    Image.fromarray(vis_image).save(buf, format="PNG")
    st.session_state["archive"].append({
        "ts":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fichier":     filename,
        "taux_%":      round(results["void_ratio"], 2),
        "plus_gros_%": round(results["largest_void_ratio"], 2),
        "nb_voids":    results["num_voids"],
        "img_bytes":   buf.getvalue(),
    })

def show_archive():
    st.subheader("🗄️ Archive des résultats")
    archive = st.session_state.get("archive", [])
    if not archive:
        st.info("Aucun résultat archivé. Cliquez sur **'📥 Archiver'** après une analyse.")
        return
    col_dl, col_vide = st.columns([2,1])
    with col_vide:
        if st.button("🗑️ Vider tous les résultats", use_container_width=True):
            st.session_state["archive"] = []
            st.rerun()
    with col_dl:
        rows = [{k:v for k,v in e.items() if k!="img_bytes"} for e in archive]
        csv  = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Exporter CSV", csv, "archive_voids.csv",
                           "text/csv", use_container_width=True)
    st.divider()
    def bc(v,t1,t2): return "🟢" if v<t1 else ("🟡" if v<t2 else "🔴")
    for i, e in enumerate(archive):
        ci, cd, cdl = st.columns([1,3,1])
        with ci:
            with st.expander("🔍 Voir", expanded=False):
                st.image(e["img_bytes"], use_container_width=True)
        with cd:
            st.markdown(
                f"**{e['fichier']}** &nbsp;·&nbsp; `{e['ts']}`\n\n"
                f"| Taux global | Plus gros void | Nb voids |\n"
                f"|:-----------:|:--------------:|:--------:|\n"
                f"| {bc(e['taux_%'],5,15)} **{e['taux_%']}%** "
                f"| {bc(e['plus_gros_%'],2,5)} **{e['plus_gros_%']}%** "
                f"| {e['nb_voids']} |"
            )
        with cdl:
            st.download_button("📥 PNG", e["img_bytes"],
                               f"analyse_{e['fichier']}", "image/png",
                               use_container_width=True, key=f"dl_{i}")
        st.divider()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    init_archive()
    st.markdown('<h1 class="main-title">🔬 Analyse RX – Détection de Voids</h1>',
                unsafe_allow_html=True)

    # On passe l'image de référence à la sidebar pour le preview live
    img_ref = st.session_state.get("img_ref_for_preview", None)
    contrast, brightness, clahe_clip, clahe_grid, sharpen, filter_geo, void_thr = sidebar(img_ref)

    tab_a, tab_arch, tab_h = st.tabs(["📤 Analyse", "🗄️ Archive", "ℹ️ Instructions"])

    # ══ ANALYSE ═══════════════════════════════════════════════════════════════
    with tab_a:

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

        raw       = np.frombuffer(up_img.read(), np.uint8)
        img_bgr   = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Stocker pour preview sidebar live
        st.session_state["img_ref_for_preview"] = image_rgb

        # 2. Masque
        mask = mask_panel(image_rgb)
        if mask is None:
            st.stop()

        # 3. Analyse
        st.subheader("3️⃣ Lancer l'analyse")
        if st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyse en cours…"):
                vis_image, results, pred_raw, proc_img = process_image(
                    image_rgb, mask, model,
                    contrast, brightness, clahe_clip, clahe_grid, sharpen,
                    filter_geo, void_thr
                )
            st.session_state["results"]   = results
            st.session_state["vis_image"] = vis_image
            st.session_state["pred_raw"]  = pred_raw
            st.session_state["proc_img"]  = proc_img
            st.session_state["last_fname"]= up_img.name

        # 4. Résultats
        if "results" in st.session_state:
            results   = st.session_state["results"]
            vis_image = st.session_state["vis_image"]
            pred_raw  = st.session_state.get("pred_raw")
            proc_img  = st.session_state.get("proc_img")
            fname     = st.session_state.get("last_fname","image.png")

            st.success("✅ Analyse terminée!")
            st.subheader("4️⃣ Résultats")

            tab_vis, tab_pre, tab_heat = st.tabs(
                ["🖼️ Analyse", "🔬 Prétraitement", "🌡️ Heatmap"])

            # ── Vue Analyse ───────────────────────────────────────────────────
            with tab_vis:
                show_color_legend()
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("**Image originale**")
                    st.image(image_rgb, use_container_width=True)
                with c2:
                    st.markdown("**Image analysée**")
                    st.image(vis_image, use_container_width=True)

            # ── Vue Prétraitement ─────────────────────────────────────────────
            with tab_pre:
                st.markdown("**Image après prétraitement (entrée du modèle)**")
                st.caption("Comparez avec l'originale : les voids sombres "
                           "devraient être plus distincts de la soudure claire.")
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("*Originale*")
                    gray_disp = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                    st.image(gray_disp, use_container_width=True, clamp=True)
                with c2:
                    st.markdown("*Après prétraitement*")
                    if proc_img is not None:
                        st.image(proc_img, use_container_width=True, clamp=True)

            # ── Heatmap ───────────────────────────────────────────────────────
            with tab_heat:
                show_heatmap_legend()
                if pred_raw is not None:
                    hc1,hc2,hc3 = st.columns(3)
                    specs = [
                        (hc1, "Canal 0 — Soudure",       cv2.COLORMAP_BONE,   0),
                        (hc2, "Canal 1 — Voids/Manques",  cv2.COLORMAP_HOT,    1),
                        (hc3, "Canal 2 — Fond",           cv2.COLORMAP_WINTER, 2),
                    ]
                    for col, label, cmap, ch in specs:
                        with col:
                            st.markdown(f"**{label}**")
                            ch_u8 = (pred_raw[:,:,ch]*255).astype(np.uint8)
                            hm    = cv2.applyColorMap(ch_u8, cmap)
                            hm    = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
                            bg    = cv2.resize(image_rgb,(hm.shape[1],hm.shape[0]))
                            blend = cv2.addWeighted(bg,0.35,hm,0.65,0)
                            st.image(blend, use_container_width=True)
                            v = pred_raw[:,:,ch]
                            st.caption(f"min={v.min():.3f} · max={v.max():.3f} "
                                       f"· moy={v.mean():.3f}")

            # ── Tableau métriques ─────────────────────────────────────────────
            st.subheader("📊 Résultats de l'analyse")
            vr = results["void_ratio"]
            lr = results["largest_void_ratio"]
            nv = results["num_voids"]

            def status(v,t1,t2):
                return "✅ Bon" if v<t1 else ("⚠️ Acceptable" if v<t2 else "❌ Non conforme")

            thr_used = results.get("void_threshold_used", 0.30)
            df = pd.DataFrame([
                {"Métrique":"Taux de manque global",       "Valeur":f"{vr:.2f}%",
                 "Seuil conforme":"< 5%","Seuil acceptable":"< 15%",
                 "Statut":status(vr,5,15)},
                {"Métrique":"Plus gros void (intérieur)",  "Valeur":f"{lr:.2f}%",
                 "Seuil conforme":"< 2%","Seuil acceptable":"< 5%",
                 "Statut":status(lr,2,5)},
                {"Métrique":"Nombre de voids détectés",    "Valeur":str(nv),
                 "Seuil conforme":"—","Seuil acceptable":"—","Statut":"ℹ️"},
                {"Métrique":"Surface inspectée",
                 "Valeur":f"{results['total_inspection_area']:,} px",
                 "Seuil conforme":"—","Seuil acceptable":"—","Statut":"ℹ️"},
                {"Métrique":"Surface voids",
                 "Valeur":f"{results['voids_area']:,} px",
                 "Seuil conforme":"—","Seuil acceptable":"—","Statut":"ℹ️"},
                {"Métrique":"Seuil void utilisé (adaptatif)",
                 "Valeur":f"{thr_used:.3f}",
                 "Seuil conforme":"—","Seuil acceptable":"—","Statut":"ℹ️"},
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Badges
            def badge(v,t1,t2):
                return ("ok","✅") if v<t1 else (("warn","⚠️") if v<t2 else ("bad","❌"))
            cg,ig = badge(vr,5,15); cl,il = badge(lr,2,5)
            m1,m2,m3 = st.columns(3)
            with m1: st.markdown(f'<div class="alert-box {cg}"><b>{ig} Taux global</b><br>'
                                 f'<span style="font-size:1.8rem;font-weight:700">{vr:.2f}%</span></div>',
                                 unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="alert-box {cl}"><b>{il} Plus gros void</b><br>'
                                 f'<span style="font-size:1.8rem;font-weight:700">{lr:.2f}%</span></div>',
                                 unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="alert-box info"><b>📍 Nb voids</b><br>'
                                 f'<span style="font-size:1.8rem;font-weight:700">{nv}</span></div>',
                                 unsafe_allow_html=True)

            # Actions
            st.subheader("💾 Actions")
            a1,a2,a3 = st.columns(3)
            with a1:
                buf = io.BytesIO()
                Image.fromarray(vis_image).save(buf, format="PNG")
                st.download_button("📥 Image analysée (PNG)", buf.getvalue(),
                                   f"analyse_{fname}", "image/png",
                                   use_container_width=True)
            with a2:
                rpt = {"fichier":fname,"ts":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "taux_%":round(vr,2),"plus_gros_%":round(lr,2),"nb_voids":nv}
                st.download_button("📥 Rapport JSON", json.dumps(rpt,indent=2),
                                   f"rapport_{fname}.json","application/json",
                                   use_container_width=True)
            with a3:
                if st.button("📥 Archiver ce résultat", use_container_width=True,
                             type="secondary"):
                    archive_result(fname, results, vis_image)
                    st.success("✅ Archivé ! → onglet 🗄️ Archive")

    # ══ ARCHIVE ═══════════════════════════════════════════════════════════════
    with tab_arch:
        show_archive()

    # ══ INSTRUCTIONS ══════════════════════════════════════════════════════════
    with tab_h:
        st.markdown("""
## 📖 Guide d'utilisation

### 1. Charger le modèle *(barre latérale)*
Fichier `.h5` issu de l'entraînement Colab → **Initialiser**.

### 2. Charger l'image RX
PNG, JPG ou JPEG.

### 3. Charger et ajuster le masque PNG
Format : **Vert** `(0,255,0)` = inspecté · **Noir** = exclu.
Les trous noirs à l'intérieur du vert excluent des zones précises (ex: billes BGA).

| Slider | Rôle |
|--------|------|
| X, Y | Décalage en % de la taille image |
| Angle | Rotation du masque |
| Échelle | 1.0 = taille originale du PNG |

### 4. Paramètres de prétraitement *(barre latérale)*

| Paramètre | Conseil |
|-----------|---------|
| **Contraste** | 1.0–1.3 pour la plupart des images |
| **Luminosité** | Ajustez si l'image est sur/sous-exposée |
| **CLAHE – Clip** | ⭐ 3–6 pour révéler les voids. Paramètre le plus important |
| **CLAHE – Grille** | 8 par défaut. Réduire à 4 pour effet très local |
| **Netteté** | 0.3–0.6 si les bords sont flous |

La **prévisualisation live** en bas de la sidebar se met à jour à chaque changement.

### 5. Onglets de résultats

| Onglet | Contenu |
|--------|---------|
| 🖼️ Analyse | Image originale vs analysée avec légende des couleurs |
| 🔬 Prétraitement | Comparaison avant/après prétraitement |
| 🌡️ Heatmap | Probabilités brutes du modèle par canal |

### 6. Interprétation couleurs
| Couleur | Signification |
|---------|--------------|
| 🔵 Bleu foncé | Soudure (canal 0 > 50%) |
| 🔴 Rouge | Void / manque (canal 1 > 50%) |
| 🟦 Cadre bleu ciel | Plus gros void intérieur |
| ⬛ Noir | Zone exclue |

### 7. Archive
**Archiver** → stocke image + métriques en session.
**Exporter CSV** → télécharge le tableau complet.
**Vider** → repart à zéro.
        """)

if __name__ == "__main__":
    main()
