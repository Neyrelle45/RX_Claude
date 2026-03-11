import sys
import subprocess
import os
import io
import tempfile
import json
from datetime import datetime

# VERSION 2024-03-06-v2 - Fix ratio calculations
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd

# ─── Imports utilitaires (détection 100% classique, sans TensorFlow) ──────────
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# VERSION 2026-03-12-v3 - OVERWRITE void_analysis_utils.py with all fixes
from utils.void_detection_V2 import (
    preprocess_image, apply_mask, analyze_voids,
    create_visualization, resize_with_aspect_ratio,
    remove_padding_and_restore, detect_voids_threshold,
    smart_add_void
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
                        contrast: float = 1.0, brightness: int = 0,
                        clahe_clip: float = 0, clahe_grid: int = 8,
                        sharpen: float = 0.3) -> np.ndarray:
    """
    Prétraitement léger pour preview visuel uniquement.
    La normalisation pour la détection des voids est faite
    automatiquement dans detect_voids_threshold (percentile robuste).

    1. Contraste/luminosité linéaires
    2. Masque de netteté optionnel
    """
    img = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    if sharpen > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img     = cv2.addWeighted(img, 1 + sharpen, blurred, -sharpen, 0)
        img     = np.clip(img, 0, 255).astype(np.uint8)
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
def process_image(image_rgb, mask_color,
                  contrast, brightness, sharpen,
                  filter_geo, sensitivity=0, min_void_px=100):
    """Analyse 100% classique — aucun modèle requis."""
    H_img, W_img = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Masque binaire
    if mask_color.ndim == 3:
        bin_mask = ((mask_color[:,:,1]>100) & (mask_color[:,:,2]<100) &
                    (mask_color[:,:,0]<100)).astype(np.uint8)
    else:
        bin_mask = (mask_color>127).astype(np.uint8)
    if bin_mask.shape != (H_img, W_img):
        bin_mask = cv2.resize(bin_mask,(W_img,H_img),interpolation=cv2.INTER_NEAREST)
        bin_mask = (bin_mask>0).astype(np.uint8)

    # 2. Prétraitement visuel léger (contraste/netteté pour preview)
    processed = preprocess_advanced(gray, contrast, brightness, sharpen=sharpen)

    # 3. Analyse avec normalisation robuste interne
    results   = analyze_voids(None, bin_mask,
                              gray_image=gray,   # image brute : normalisation auto interne
                              sensitivity=sensitivity,
                              min_void_px=min_void_px)
    vis_image = create_visualization(image_rgb, None, bin_mask, results)
    return vis_image, results, processed

# ─── Preview prétraitement live ───────────────────────────────────────────────
def preprocess_preview(image_rgb, contrast, brightness, sharpen):
    """Retourne l'image prétraitée pour preview live."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    proc = preprocess_advanced(gray, contrast, brightness, sharpen=sharpen)
    return cv2.cvtColor(proc, cv2.COLOR_GRAY2RGB)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
def sidebar(image_rgb_ref):
    """
    Retourne les paramètres. Si image_rgb_ref fournie, affiche preview live.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Prétraitement - compact
        st.markdown("### 🎛️ Prétraitement")
        with st.expander("ℹ️ Guide", expanded=False):
            st.markdown("""
**Contraste** : amplifie les niveaux de gris.
**Luminosité** : décale vers clair/sombre.
**Netteté** : accentue les bords.
            """)

        st.caption("Contraste"); contrast = st.slider("c1", 0.5, 2.0, 1.0, 0.05, 
                                                       label_visibility="collapsed")
        st.caption("Luminosité"); brightness = st.slider("c2", -50, 50, 0, 5,
                                                          label_visibility="collapsed")
        st.caption("Netteté"); sharpen = st.slider("c3", 0.0, 2.0, 0.3, 0.1,
                                                    label_visibility="collapsed")
        clahe_clip = 0.0; clahe_grid = 8

        # Preview compact
        if image_rgb_ref is not None:
            with st.expander("👁️ Preview", expanded=False):
                prev = preprocess_preview(image_rgb_ref, contrast, brightness, sharpen)
                st.image(prev, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔍 Analyse")
        filter_geo = st.checkbox("Filtrer géométrie", value=True,
                                help="Exclut cercles/rectangles parfaits")

        st.markdown("---")
        st.markdown("### 🎯 Détection")
        st.caption("Ajustement seuil")
        sensitivity = st.slider("sens", -30, 30, 0, 5, label_visibility="collapsed",
            help="Biais par défaut = -10 (agressif). 0 = biais -10, +10 = Otsu pur, -10 = très agressif")
        if sensitivity == 0:
            st.caption("💡 Voids manqués ? Laissez à 0 (biais -10). Trop de détection ? → +10 à +15")
        st.caption("Taille min. void (px)")
        min_void_px = st.slider("minv", 10, 1000, 20, 10, label_visibility="collapsed",
            help="Blobs plus petits ignorés. 20px = défaut (taille significative).")
        solder_thr = None   # non utilisé dans approche classique

    return contrast, brightness, sharpen, filter_geo, sensitivity, min_void_px

# ─── MASQUE ───────────────────────────────────────────────────────────────────
def mask_panel(image_rgb, uploaded_mask_raw):
    """Ajustement du masque uploadé (sans uploader ici)."""
    if uploaded_mask_raw is None:
        return None
    
    H, W = image_rgb.shape[:2]
    mask_raw = uploaded_mask_raw
    
    if mask_raw.max() == 0:
        st.error("❌ Masque vide")
        return None

    st.markdown("### 2️⃣ Ajustement masque")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.caption("X"); tx = st.slider("tx", -50, 50, 0, 1, label_visibility="collapsed")
    with c2: st.caption("Y"); ty = st.slider("ty", -50, 50, 0, 1, label_visibility="collapsed")
    with c3: st.caption("Angle"); angle = st.slider("ang", -180, 180, 0, 1, label_visibility="collapsed")
    with c4: st.caption("Échelle"); scale = st.slider("sc", 0.1, 3.0, 1.0, 0.01, label_visibility="collapsed")

    cr, ci = st.columns([1,3])
    with cr:
        if st.button("↺ Reset", use_container_width=True):
            for k in ["tx","ty","ang"]: 
                if k in st.session_state: st.session_state[k]=0
            if "sc" in st.session_state: st.session_state["sc"]=1.0
            st.rerun()
    with ci:
        pct_src = mask_raw.mean()/255*100
        st.caption(f"Source : {mask_raw.shape[1]}×{mask_raw.shape[0]} px · {pct_src:.1f}% vert")

    mask_color = transform_mask(mask_raw, H, W, tx, ty, scale, angle)
    pct = (mask_color[:,:,1]>100).mean()*100
    if pct < 0.5:
        st.warning("⚠️ Masque hors image")

    cp1, cp2 = st.columns(2)
    with cp1:
        st.image(overlay_preview(image_rgb, mask_color),
                 caption=f"Preview — {pct:.1f}% inspecté",
                 use_container_width=True)
    with cp2:
        disp = np.zeros((H,W,3), dtype=np.uint8)
        disp[:,:,1] = mask_color[:,:,1]
        st.image(disp, caption="Masque seul", use_container_width=True)

    return mask_color

def show_color_legend():
    st.markdown("""
<div class="legend-box">
  <div class="legend-item">
    <div class="swatch" style="background:#22aa44"></div>
    <span><b>Vert</b> — Soudure présente (zone inspectée sans manque)</span>
  </div>
  <div class="legend-item">
    <div class="swatch" style="background:#e61414"></div>
    <span><b>Rouge vif</b> — Void / manque de soudure (zone la plus sombre dans la soudure)</span>
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
    st.markdown("---")
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
        st.markdown("---")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    init_archive()
    st.markdown('<h1 class="main-title">🔬 Analyse RX – Détection de Voids</h1>',
                unsafe_allow_html=True)

    # On passe l'image de référence à la sidebar pour le preview live
    img_ref = st.session_state.get("img_ref_for_preview", None)
    contrast, brightness, sharpen, filter_geo, sensitivity, min_void_px = sidebar(img_ref)

    tab_a, tab_arch, tab_h = st.tabs(["📤 Analyse", "🗄️ Archive", "ℹ️ Instructions"])

    # ══ ANALYSE ═══════════════════════════════════════════════════════════════
    with tab_a:

        # Uploaders côte à côte + ordre flexible
        st.markdown("### 📥 Charger les fichiers")
        _c1, _c2 = st.columns(2)
        with _c1:
            up_img = st.file_uploader("Image RX", type=["png","jpg","jpeg"], 
                                     key="up_img")
        with _c2:
            up_mask = st.file_uploader("Masque (optionnel)", 
                                      type=["png","jpg","jpeg"], key="up_mask")

        # Charger masque (si fourni, sinon dessin manuel plus tard)
        mask = None
        if up_mask is not None:
            raw_m = np.frombuffer(up_mask.read(), np.uint8)
            mask_bgr = cv2.imdecode(raw_m, cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
            st.session_state["uploaded_mask"] = mask
            st.success(f"✅ Masque chargé : {mask.shape[1]}×{mask.shape[0]} px")
        elif "uploaded_mask" in st.session_state:
            mask = st.session_state["uploaded_mask"]
            st.info("ℹ️ Masque précédent conservé")

        # Charger image RX
        if up_img is None:
            st.info("👆 Chargez une image RX pour commencer")
            st.stop()

        raw = np.frombuffer(up_img.read(), np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.session_state["img_ref_for_preview"] = image_rgb

        # Traiter le masque avec l'interface d'ajustement
        if mask is not None:
            # Décoder si c'est RGB
            if mask.ndim == 3:
                mask_gray = ((mask[:,:,1] > 100) & (mask[:,:,2] < 100) & 
                            (mask[:,:,0] < 100)).astype(np.uint8) * 255
            else:
                mask_gray = (mask > 127).astype(np.uint8) * 255
            
            mask = mask_panel(image_rgb, mask_gray)
            if mask is None:
                st.stop()
        else:
            st.error("❌ Chargez un masque PNG pour continuer")
            st.stop()

        # 3. Analyse
        st.markdown("### 3️⃣ Analyse")
        if st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyse en cours…"):
                vis_image, results, proc_img = process_image(
                    image_rgb, mask,
                    contrast, brightness, sharpen,
                    filter_geo, sensitivity, min_void_px
                )
            st.session_state["results"]   = results
            st.session_state["vis_image"] = vis_image
            st.session_state["pred_raw"]  = None
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
            
            # Afficher debug info si 0 void détecté
            debug = results.get("debug_info", {})
            if debug and results.get("num_voids", 0) == 0:
                with st.expander("🔍 Debug : Pourquoi 0 void détecté ?", expanded=True):
                    st.warning("⚠️ Aucun void détecté — voici le diagnostic :")
                    st.write(f"**Pixels détectés (Otsu brut)** : {debug.get('pixels_bruts', 0):,} px")
                    st.write(f"**Après morphologie** : {debug.get('pixels_morph', 0):,} px")
                    st.write(f"**Blobs avant filtrage** : {debug.get('blobs_avant', 0)}")
                    st.write(f"**Blobs après filtrage** : {debug.get('blobs_apres', 0)}")
                    if debug.get('rejets'):
                        st.write(f"**Blobs rejetés** ({len(debug['rejets'])}) :")
                        for rejet in debug['rejets'][:10]:  # Max 10 premiers
                            st.caption(f"  • {rejet}")
                    st.info("💡 **Solutions** : Ajustez Sensibilité (+10 à +20) ou Taille min (20px)")
            
            st.markdown("### 4️⃣ Résultats")

            tab_vis, tab_pre, tab_cumul = st.tabs(
                ["🖼️ Analyse", "🔬 Prétraitement", "📊 Cumul résultats"])

            # ── Vue Analyse ───────────────────────────────────────────────────
            with tab_vis:
                show_color_legend()
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown("**Image originale**")
                    st.image(image_rgb, use_container_width=True)
                with c2:
                    st.markdown("**Image analysée**")
                    # Toujours lire depuis session_state pour refléter les corrections
                    st.image(st.session_state["vis_image"], use_container_width=True)

                # ── Correction manuelle par clic AUTOMATIQUE ──────────────────
                st.markdown("---")
                st.markdown("**✏️ Correction manuelle — cliquez directement sur l'image**")

                void_mask_edit = st.session_state["results"].get("void_mask")
                if void_mask_edit is not None:
                    if "manual_overrides" not in st.session_state:
                        st.session_state["manual_overrides"] = []

                    _H_nat, _W_nat = void_mask_edit.shape[:2]
                    _DISP_W = 700
                    _DISP_H = int(_H_nat * _DISP_W / max(_W_nat, 1))

                    ov_action = st.radio(
                        "Mode :",
                        ["❌ Supprimer void (clic rouge→vert)",
                         "✅ Ajouter void  (clic vert→rouge)"],
                        horizontal=True, key="ov_action")

                    # Image cliquable - CROP + ZOOM sur ROI avec offset tracking
                    from PIL import Image as _PIL2
                    # Recréer visualization NON croppée d'abord
                    _bm_manual = ((mask[:,:,1]>100)&(mask[:,:,2]<100)&
                                  (mask[:,:,0]<100)).astype(np.uint8) \
                                 if mask.ndim==3 else (mask>127).astype(np.uint8)
                    _vis_full = create_visualization(
                        image_rgb, None, _bm_manual, 
                        st.session_state["results"], 
                        no_crop=True  # Full size pour calculs
                    ).astype(np.uint8)
                    
                    # Calculer bbox de la ROI pour crop + zoom
                    if _bm_manual.any():
                        ys, xs = np.where(_bm_manual > 0)
                        if len(ys) > 0:
                            y_min, y_max = int(ys.min()), int(ys.max()) + 1
                            x_min, x_max = int(xs.min()), int(xs.max()) + 1
                            # Marge 10px
                            y_min = max(0, y_min - 10)
                            y_max = min(_H_nat, y_max + 10)
                            x_min = max(0, x_min - 10)
                            x_max = min(_W_nat, x_max + 10)
                            # Crop sur ROI
                            _vis_cropped = _vis_full[y_min:y_max, x_min:x_max]
                            _crop_offset_x = x_min
                            _crop_offset_y = y_min
                            _crop_w = x_max - x_min
                            _crop_h = y_max - y_min
                        else:
                            _vis_cropped = _vis_full
                            _crop_offset_x = _crop_offset_y = 0
                            _crop_w, _crop_h = _W_nat, _H_nat
                    else:
                        _vis_cropped = _vis_full
                        _crop_offset_x = _crop_offset_y = 0
                        _crop_w, _crop_h = _W_nat, _H_nat
                    
                    # Resize pour affichage
                    _vis_pil = _PIL2.fromarray(_vis_cropped).resize(
                        (_DISP_W, _DISP_H), _PIL2.LANCZOS)

                    _coords = None
                    try:
                        from streamlit_image_coordinates import streamlit_image_coordinates
                        # streamlit_image_coordinates affiche toujours l'image
                        _coords = streamlit_image_coordinates(_vis_pil, key="image_coords")
                    except (ImportError, Exception):
                        # Si le package n'existe pas, afficher l'image statique
                        st.image(_vis_pil, use_container_width=False, width=_DISP_W,
                                caption="📍 Package streamlit-image-coordinates manquant")
                        _coords = None

                    # ── ACTION AU CLIC (avec offset correction) ───────────────
                    if _coords is not None:
                        _x_disp = _coords.get("x", 0)
                        _y_disp = _coords.get("y", 0)
                        # Convertir coords display → coords dans l'image croppée
                        _x_crop = int(np.clip(_x_disp * _crop_w / _DISP_W, 0, _crop_w - 1))
                        _y_crop = int(np.clip(_y_disp * _crop_h / _DISP_H, 0, _crop_h - 1))
                        # Ajouter offset pour obtenir coords dans l'image NATIVE
                        ov_x = int(np.clip(_x_crop + _crop_offset_x, 0, _W_nat - 1))
                        ov_y = int(np.clip(_y_crop + _crop_offset_y, 0, _H_nat - 1))
                        
                        # Éviter boucle : exécuter SEULEMENT si nouveau clic
                        _action_key = f"{ov_x}_{ov_y}_{ov_action}"
                        _last_action = st.session_state.get("last_action_executed", "")
                        
                        if _action_key != _last_action:
                            # NOUVEAU CLIC → marquer ET exécuter
                            st.session_state["last_action_executed"] = _action_key
                        
                            # Exécuter l'action
                            from skimage import measure as _meas2
                            void_now = st.session_state["results"]["void_mask"]
                            _bm2 = ((mask[:,:,1]>100)&(mask[:,:,2]<100)&
                                    (mask[:,:,0]<100)).astype(np.uint8) \
                                   if mask.ndim==3 else (mask>127).astype(np.uint8)

                            if "Supprimer" in ov_action:
                                _lab = _meas2.label(void_now.astype(np.uint8), connectivity=2)
                                _bid = int(_lab[ov_y, ov_x])
                                if _bid > 0:
                                    _bpx = (_lab == _bid)
                                    void_now[_bpx] = False
                                    st.session_state["results"]["void_mask"] = void_now
                                    st.session_state["manual_overrides"].append({"a":"rm"})
                                    st.session_state["vis_image"] = create_visualization(
                                        image_rgb, None, _bm2, st.session_state["results"])
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ Pas de void au pixel ({ov_x},{ov_y})")
                            else:
                                _gray_raw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                                _nv, _n_added = smart_add_void(_gray_raw, _bm2, void_now, ov_y, ov_x)
                                if _n_added > 0:
                                    st.session_state["results"]["void_mask"] = _nv
                                    st.session_state["manual_overrides"].append({"a":"add"})
                                    st.session_state["vis_image"] = create_visualization(
                                        image_rgb, None, _bm2, st.session_state["results"])
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ Pas de zone claire au pixel ({ov_x},{ov_y})")
                    
                    # Bouton reset seul
                    do_reset = st.button("🔄 Réinitialiser toutes les corrections",
                                        use_container_width=True, type="secondary")


                    if do_reset:
                        st.session_state["manual_overrides"] = []
                        _vr,_rs,_pi = process_image(
                            image_rgb, mask, contrast, brightness,
                            sharpen, filter_geo, sensitivity, min_void_px)
                        st.session_state["results"]   = _rs
                        st.session_state["vis_image"] = _vr
                        st.session_state["proc_img"]  = _pi
                        st.rerun()

                    if st.session_state["manual_overrides"]:
                        n=len(st.session_state["manual_overrides"])
                        st.caption(f"📝 {n} correction(s) — réinitialisez pour revenir à l'analyse brute")

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

            # ── Cumul résultats ───────────────────────────────────────────
            with tab_cumul:
                st.markdown("### 📊 Cumul des résultats archivés")
                _archive = st.session_state.get("archive", [])
                if not _archive:
                    st.info("Aucun résultat archivé. Lancez une analyse puis cliquez **📥 Archiver**.")
                else:
                    # Tableau récapitulatif
                    _rows = [{k:v for k,v in e.items() if k != "img_bytes"}
                             for e in _archive]
                    _df_cumul = pd.DataFrame(_rows)
                    # Colonnes renommées lisiblement
                    _df_cumul = _df_cumul.rename(columns={
                        "fichier":"Fichier","ts":"Horodatage",
                        "taux_%":"Taux global %","plus_gros_%":"Plus gros void %",
                        "nb_voids":"Nb voids"})
                    st.dataframe(_df_cumul, use_container_width=True, hide_index=True)

                    # Graphique tendance si >= 2 entrées
                    if len(_archive) >= 2:
                        st.markdown("#### Évolution du taux de manque")
                        _taux   = [e["taux_%"]    for e in _archive]
                        _labels = [e["fichier"]   for e in _archive]
                        _chart  = pd.DataFrame({
                            "Image": _labels,
                            "Taux global (%)":    [e["taux_%"]     for e in _archive],
                            "Plus gros void (%)": [e["plus_gros_%"] for e in _archive],
                        }).set_index("Image")
                        st.line_chart(_chart)

                    # Statistiques globales
                    st.markdown("#### Statistiques globales")
                    _taux_vals = [e["taux_%"] for e in _archive]
                    _gros_vals = [e["plus_gros_%"] for e in _archive]
                    _sc1,_sc2,_sc3,_sc4 = st.columns(4)
                    _sc1.metric("Moyenne taux",   f"{sum(_taux_vals)/len(_taux_vals):.2f}%")
                    _sc2.metric("Max taux",        f"{max(_taux_vals):.2f}%")
                    _sc3.metric("Moy. gros void",  f"{sum(_gros_vals)/len(_gros_vals):.2f}%")
                    _sc4.metric("Images analysées",str(len(_archive)))

                    # Export CSV
                    _csv_all = pd.DataFrame(_rows).to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Exporter tout en CSV", _csv_all,
                                       "cumul_voids.csv","text/csv",
                                       use_container_width=False)

            # ── Tableau métriques ─────────────────────────────────────────────
            st.markdown("#### 📊 Métriques")
            # Recalculer les métriques depuis le void_mask courant (peut avoir été édité)
            _cur_vm = st.session_state["results"].get("void_mask")
            if _cur_vm is not None and _cur_vm.any():
                from skimage import measure as _meas_m
                # Surface totale inspectée (fixe, ne change pas avec les corrections)
                _n_total = int(st.session_state["results"].get("total_inspection_area", 1))
                _n_v     = int(_cur_vm.sum())
                _vr_live = _n_v / max(_n_total, 1) * 100
                
                # Recalculer le plus gros void (simplement le blob avec la plus grande aire)
                _lbl_m   = _meas_m.label(_cur_vm.astype(np.uint8), connectivity=2)
                _big_area = 0
                for _rm in _meas_m.regionprops(_lbl_m):
                    if _rm.area > _big_area:
                        _big_area = _rm.area
                _lr_live = _big_area / max(_n_total, 1) * 100 if _big_area > 0 else 0.0
                _nv_live = int(_lbl_m.max())
            else:
                _vr_live = st.session_state["results"].get("void_ratio", 0)
                _lr_live = st.session_state["results"].get("largest_void_ratio", 0)
                _nv_live = st.session_state["results"].get("num_voids", 0)
            vr = _vr_live; lr = _lr_live; nv = _nv_live

            # Tableau sans seuils ni jugements (varient selon clients)
            df = pd.DataFrame([
                {"Métrique":"Taux de manque global",       "Valeur":f"{vr:.2f}%"},
                {"Métrique":"Plus gros void (intérieur)",  "Valeur":f"{lr:.2f}%"},
                {"Métrique":"Nombre de voids détectés",    "Valeur":str(nv)},
                {"Métrique":"Surface inspectée",
                 "Valeur":f"{results['total_inspection_area']:,} px"},
                {"Métrique":"Surface soudure",
                 "Valeur":f"{results.get('solder_area',results.get('voids_area',0)):,} px"},
                {"Métrique":"Surface voids",
                 "Valeur":f"{results['voids_area']:,} px"},
                {"Métrique":"Sensibilité utilisée",
                 "Valeur":f"{results.get('void_threshold_used',0):.1f} px gris"},
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Badges (se mettent à jour avec corrections manuelles)
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
                    # Utiliser l'image CORRIGÉE depuis session_state
                    vis_corrected = st.session_state.get("vis_image", vis_image)
                    # Utiliser les métriques LIVE (recalculées après corrections)
                    results_corrected = st.session_state["results"].copy()
                    results_corrected["void_ratio"] = vr  # vr = _vr_live
                    results_corrected["largest_void_ratio"] = lr  # lr = _lr_live
                    results_corrected["num_voids"] = nv  # nv = _nv_live
                    archive_result(fname, results_corrected, vis_corrected)
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
| 📊 Cumul résultats | Tableau et graphes des analyses archivées |

### 6. Interprétation couleurs
| Couleur | Signification |
|---------|--------------|
| 🟢 Vert | Soudure présente (zone inspectée sans manque) |
| 🔴 Rouge | Void / manque de soudure (zone la plus sombre) |
| ⬛ Noir | Zone exclue par le masque (non analysée) |

### 7. Archive
**Archiver** → stocke image + métriques en session.
**Exporter CSV** → télécharge le tableau complet.
**Vider** → repart à zéro.
        """)

if __name__ == "__main__":
    main()
