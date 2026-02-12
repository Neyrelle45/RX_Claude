import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import io

# Import des fonctions utilitaires
from void_analysis_utils import (
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


def create_mask_from_canvas(canvas_result, original_shape):
    """Crée un masque à partir du dessin sur canvas"""
    if canvas_result.image_data is None:
        return None
    
    # Extraire le canal alpha (où les dessins apparaissent)
    canvas_data = canvas_result.image_data[:, :, 3]
    
    # Créer un masque: zones dessinées = 1, reste = 0
    mask = (canvas_data > 0).astype(np.uint8) * 255
    
    # Redimensionner au format original si nécessaire
    if mask.shape != original_shape[:2]:
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
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
        uploaded_model = st.file_uploader(
            "Charger le fichier du modèle (.h5)",
            type=['h5'],
            help="Sélectionnez le fichier .h5 du modèle entraîné"
        )
        
        if uploaded_model is not None:
            if st.button("🔄 Initialiser le modèle", use_container_width=True):
                with st.spinner("Chargement du modèle..."):
                    # Création d'un fichier temporaire car Keras a besoin d'un chemin 
                    # physique pour charger un modèle .h5 complet
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                        tmp.write(uploaded_model.getvalue())
                        tmp_path = tmp.name
                    
                    model = load_model(tmp_path)
                    
                    if model is not None:
                        st.session_state['model'] = model
                        st.success("✅ Modèle chargé avec succès!")
                        # Nettoyage du fichier temporaire
                        os.remove(tmp_path)
        
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
                # Canvas pour dessiner le masque
                canvas_height = min(600, int(image_rgb.shape[0] * 600 / image_rgb.shape[1]))
                
                canvas_result = st_canvas(
                    fill_color="rgba(0, 255, 0, 0.3)",
                    stroke_width=20,
                    stroke_color="rgba(0, 255, 0, 0.8)",
                    background_image=Image.fromarray(image_rgb),
                    update_streamlit=True,
                    height=canvas_height,
                    width=600,
                    drawing_mode="freedraw",
                    key="canvas",
                )
            
            with col2:
                st.markdown("**Outils de dessin:**")
                st.markdown("- 🖊️ Dessinez pour créer le masque")
                st.markdown("- 🗑️ Utilisez l'effaceur pour corriger")
                st.markdown("- 🔄 Rafraîchissez pour recommencer")
                
                if st.button("🗑️ Effacer le masque", use_container_width=True):
                    st.rerun()
                
                # Option de masque rectangulaire simple
                st.divider()
                st.markdown("**Masque rectangulaire:**")
                use_rect_mask = st.checkbox("Utiliser un masque rectangulaire")
                
                if use_rect_mask:
                    st.markdown("Définissez les marges (en % de l'image):")
                    margin_top = st.slider("Marge haut", 0, 50, 10)
                    margin_bottom = st.slider("Marge bas", 0, 50, 10)
                    margin_left = st.slider("Marge gauche", 0, 50, 10)
                    margin_right = st.slider("Marge droite", 0, 50, 10)
            
            # Bouton d'analyse
            st.subheader("3️⃣ Lancer l'analyse")
            
            if st.button("🚀 Analyser", type="primary", use_container_width=True):
                # Créer le masque
                if use_rect_mask:
                    # Masque rectangulaire
                    h, w = image_rgb.shape[:2]
                    mask = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    top = int(h * margin_top / 100)
                    bottom = int(h * (100 - margin_bottom) / 100)
                    left = int(w * margin_left / 100)
                    right = int(w * (100 - margin_right) / 100)
                    
                    mask[top:bottom, left:right, 1] = 255  # Vert
                else:
                    # Masque dessiné
                    if canvas_result.image_data is not None:
                        canvas_mask = create_mask_from_canvas(canvas_result, image_rgb.shape)
                        if canvas_mask is None or np.sum(canvas_mask) == 0:
                            st.error("❌ Veuillez dessiner un masque avant d'analyser.")
                            return
                        
                        # Convertir en masque couleur (vert)
                        mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)
                        mask[:, :, 1] = canvas_mask
                    else:
                        st.error("❌ Veuillez dessiner un masque avant d'analyser.")
                        return
                
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
