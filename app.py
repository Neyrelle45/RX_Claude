import sys
import subprocess
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import io
import streamlit as st

# ─── Auto-installation de TensorFlow ─────────────────────────────────────────
# TF ne peut pas être dans requirements.txt (conflit avec les dépendances
# système de Streamlit Cloud). On l'installe une seule fois ici.
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
# ─────────────────────────────────────────────────────────────────────────────

# Import des fonctions utilitaires
from utils.void_analysis_utils import (
    preprocess_image, apply_mask, analyze_voids,
    create_visualization, resize_with_aspect_ratio
)

# Configuration de la page
st.set_page_config(
    page_title="Analyse RX - Détection de Voids",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Charge le modèle TensorFlow"""
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'combined_loss': lambda y_true, y_pred: 0.5 * tf.keras.losses.categorical_crossentropy(y_true, y_pred) + 
                                                        0.5 * (1 - dice_coefficient(y_true, y_pred)),
                'dice_coefficient': dice_coefficient
            },
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Coefficient de Dice"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def create_mask_from_coords(h, w, x1_pct, y1_pct, x2_pct, y2_pct):
    """Crée un masque rectangulaire à partir de pourcentages de coordonnées"""
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    x1 = int(w * x1_pct / 100)
    y1 = int(h * y1_pct / 100)
    x2 = int(w * x2_pct / 100)
    y2 = int(h * y2_pct / 100)
    mask[y1:y2, x1:x2, 1] = 255  # vert = zone à inspecter
    return mask


def process_image(image, mask, model, contrast, brightness, filter_geometric):
    """Traite l'image et retourne les résultats"""
    
    # Convertir en niveaux de gris si nécessaire
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image.copy()
    
    # Prétraitement
    processed = preprocess_image(gray_image, contrast, brightness)
    
    # Appliquer le masque
    masked_image, inspection_mask = apply_mask(processed, mask)
    
    # Préparer pour le modèle
    input_size = (512, 512)
    resized, transform_params = resize_with_aspect_ratio(masked_image, input_size)
    
    # Normaliser et ajouter dimensions
    model_input = resized.astype(np.float32) / 255.0
    model_input = np.expand_dims(model_input, axis=(0, -1))
    
    # Prédiction
    prediction = model.predict(model_input, verbose=0)[0]
    
    # Redimensionner la prédiction à la taille originale
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
    
    # Analyser les voids
    results = analyze_voids(prediction_resized, inspection_mask, filter_geometric)
    
    # Créer la visualisation
    vis_image = create_visualization(image, prediction_resized, inspection_mask, results)
    
    return vis_image, results


def main():
    """Application principale"""
    
    # Titre
    st.markdown('<h1 class="main-title">🔬 Analyse RX - Détection de Voids</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Chargement du modèle
        st.subheader("Modèle")
        model_path = st.text_input(
            "Chemin du modèle",
            value="models/void_detection_best.h5",
            help="Chemin vers le fichier .h5 du modèle entraîné"
        )
        
        if st.button("🔄 Charger le modèle", use_container_width=True):
            with st.spinner("Chargement du modèle..."):
                model = load_model(model_path)
                if model is not None:
                    st.session_state['model'] = model
                    st.success("✅ Modèle chargé avec succès!")
        
        st.divider()
        
        # Paramètres de prétraitement
        st.subheader("Prétraitement")
        
        contrast = st.slider(
            "Contraste",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Ajuste le contraste de l'image"
        )
        
        brightness = st.slider(
            "Luminosité",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            help="Ajuste la luminosité de l'image"
        )
        
        st.divider()
        
        # Paramètres d'analyse
        st.subheader("Analyse")
        
        filter_geometric = st.checkbox(
            "Filtrer les formes géométriques",
            value=True,
            help="Exclut les formes rectangulaires et circulaires parfaites (pistes, vias)"
        )
        
        st.divider()
        
        # Paramètres de visualisation
        st.subheader("Visualisation")
        
        show_legend = st.checkbox("Afficher la légende", value=True)
    
    # Zone principale
    tab1, tab2 = st.tabs(["📤 Analyse", "ℹ️ Instructions"])
    
    with tab1:
        # Vérifier si le modèle est chargé
        if 'model' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord charger un modèle dans la barre latérale.")
            return
        
        model = st.session_state['model']
        
        # Upload de l'image
        st.subheader("1️⃣ Charger l'image RX")
        uploaded_file = st.file_uploader(
            "Sélectionnez une image",
            type=['png', 'jpg', 'jpeg'],
            help="Formats acceptés: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            # Charger l'image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convertir en RGB si nécessaire
            if len(image_array.shape) == 2:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB) if image_array.shape[2] == 4 else image_array
            
            st.session_state['original_image'] = image_rgb
            
            # Section de dessin du masque
            st.subheader("2️⃣ Dessiner le masque d'inspection")
            st.info("🖊️ Dessinez la zone d'inspection sur l'image en vert. Les zones non dessinées seront exclues de l'analyse.")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Prévisualisation de l'image
                st.image(image_rgb, caption="Image chargée", use_container_width=True)
            
            with col2:
                st.markdown("**Zone d'inspection (% de l'image):**")
                st.caption("Délimitez le rectangle de la zone à analyser.")
                x1_pct = st.slider("Bord gauche  (%)", 0, 49, 10, key="x1")
                x2_pct = st.slider("Bord droit   (%)", 51, 100, 90, key="x2")
                y1_pct = st.slider("Bord haut    (%)", 0, 49, 10, key="y1")
                y2_pct = st.slider("Bord bas     (%)", 51, 100, 90, key="y2")
                
                # Prévisualisation du masque sur l'image
                preview = image_rgb.copy()
                h_p, w_p = preview.shape[:2]
                x1p = int(w_p * x1_pct / 100); x2p = int(w_p * x2_pct / 100)
                y1p = int(h_p * y1_pct / 100); y2p = int(h_p * y2_pct / 100)
                cv2.rectangle(preview, (x1p, y1p), (x2p, y2p), (0, 255, 0), 3)
                st.image(preview, caption="Zone verte = zone inspectée",
                         use_container_width=True)
            
            # Bouton d'analyse
            st.subheader("3️⃣ Lancer l'analyse")
            
            if st.button("🚀 Analyser", type="primary", use_container_width=True):
                # Créer le masque rectangulaire depuis les sliders
                mask = create_mask_from_coords(
                    image_rgb.shape[0], image_rgb.shape[1],
                    x1_pct, y1_pct, x2_pct, y2_pct
                )
                
                # Traiter l'image
                with st.spinner("🔄 Analyse en cours..."):
                    vis_image, results = process_image(
                        image_rgb, mask, model, contrast, brightness, filter_geometric
                    )
                
                # Afficher les résultats
                st.success("✅ Analyse terminée!")
                
                # Images côte à côte
                st.subheader("4️⃣ Résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Image originale**")
                    st.image(image_rgb, use_container_width=True)
                
                with col2:
                    st.markdown("**Image analysée**")
                    if show_legend:
                        st.markdown("""
                        **Légende:**
                        - 🔵 **Bleu foncé**: Soudure
                        - 🔴 **Rouge**: Voids/Manques
                        - 🟦 **Bleu ciel**: Plus gros void
                        """)
                    st.image(vis_image, use_container_width=True)
                
                # Tableau de résultats
                st.subheader("📊 Statistiques")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    void_ratio = results['void_ratio']
                    if void_ratio < 5:
                        status_class = "alert-success"
                        status_icon = "✅"
                    elif void_ratio < 15:
                        status_class = "alert-warning"
                        status_icon = "⚠️"
                    else:
                        status_class = "alert-danger"
                        status_icon = "❌"
                    
                    st.markdown(f"""
                    <div class="alert-box {status_class}">
                        <h3>{status_icon} Taux de manque global</h3>
                        <h2>{void_ratio:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    largest_ratio = results['largest_void_ratio']
                    if largest_ratio < 2:
                        status_class = "alert-success"
                        status_icon = "✅"
                    elif largest_ratio < 5:
                        status_class = "alert-warning"
                        status_icon = "⚠️"
                    else:
                        status_class = "alert-danger"
                        status_icon = "❌"
                    
                    st.markdown(f"""
                    <div class="alert-box {status_class}">
                        <h3>{status_icon} Plus gros void</h3>
                        <h2>{largest_ratio:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="alert-box alert-box" style="background-color: #e7f3ff; border: 1px solid #b3d9ff;">
                        <h3>📍 Nombre de voids</h3>
                        <h2>{results['num_voids']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Détails supplémentaires
                with st.expander("📋 Détails complets"):
                    st.json({
                        "Surface totale inspectée (pixels)": int(results['total_inspection_area']),
                        "Surface de soudure (pixels)": int(results['soudure_area']),
                        "Surface de voids (pixels)": int(results['voids_area']),
                        "Taux de voids (%)": round(results['void_ratio'], 2),
                        "Surface du plus gros void (pixels)": int(results['largest_void_area']),
                        "Ratio du plus gros void (%)": round(results['largest_void_ratio'], 2),
                        "Nombre total de voids": results['num_voids']
                    })
                
                # Téléchargement des résultats
                st.subheader("💾 Télécharger")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Télécharger l'image analysée
                    vis_pil = Image.fromarray(vis_image)
                    buf = io.BytesIO()
                    vis_pil.save(buf, format='PNG')
                    btn = st.download_button(
                        label="📥 Télécharger l'image analysée",
                        data=buf.getvalue(),
                        file_name="analyse_voids.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col2:
                    # Télécharger le rapport JSON
                    report = {
                        "taux_manque_global_%": round(results['void_ratio'], 2),
                        "taux_plus_gros_void_%": round(results['largest_void_ratio'], 2),
                        "nombre_voids": results['num_voids'],
                        "surface_inspection_pixels": int(results['total_inspection_area']),
                        "surface_soudure_pixels": int(results['soudure_area']),
                        "surface_voids_pixels": int(results['voids_area'])
                    }
                    
                    btn = st.download_button(
                        label="📥 Télécharger le rapport JSON",
                        data=str(report),
                        file_name="rapport_analyse.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    with tab2:
        st.markdown("""
        ## 📖 Guide d'utilisation
        
        ### 1. Chargement du modèle
        - Dans la barre latérale, entrez le chemin vers votre modèle entraîné (fichier `.h5`)
        - Cliquez sur "Charger le modèle"
        - Attendez la confirmation
        
        ### 2. Analyse d'une image
        1. **Charger l'image**: Uploadez votre cliché rayon X
        2. **Dessiner le masque**: Définissez la zone d'inspection
           - Mode libre: dessinez directement sur l'image
           - Mode rectangulaire: définissez des marges en pourcentages
        3. **Ajuster les paramètres** (optionnel):
           - Contraste et luminosité
           - Filtrage des formes géométriques
        4. **Lancer l'analyse**
        
        ### 3. Interprétation des résultats
        
        #### Visualisation
        - **Bleu foncé**: Zones de soudure détectées
        - **Rouge**: Voids et manques de soudure
        - **Bleu ciel (cadre épais)**: Le plus gros void détecté
        
        #### Métriques
        - **Taux de manque global**: Pourcentage de voids par rapport à la zone inspectée
        - **Plus gros void**: Taille du void le plus important (excluant ceux touchant les bords)
        - **Nombre de voids**: Total de défauts détectés
        
        #### Seuils de qualité
        - ✅ **Bon**: Taux < 5%
        - ⚠️ **Acceptable**: Taux entre 5-15%
        - ❌ **Non conforme**: Taux > 15%
        
        ### 4. Conseils
        
        - **Masque d'inspection**: Dessinez précisément la zone à analyser pour éviter les faux positifs
        - **Contraste**: Augmentez si les voids sont peu visibles
        - **Filtrage géométrique**: Activé par défaut pour exclure les éléments du PCB (pistes, vias)
        - **Images multiples**: Utilisez l'application batch pour traiter plusieurs images
        
        ### 5. Export
        - **Image analysée**: Format PNG avec visualisation colorée
        - **Rapport JSON**: Données quantitatives pour traçabilité
        """)


if __name__ == "__main__":
    main()
