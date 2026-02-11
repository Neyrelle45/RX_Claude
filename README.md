# 📦 Modèles

Le modèle entraîné `void_detection_best.h5` est **trop volumineux** pour être hébergé sur GitHub (limite de 100 MB).

## 🎯 Comment obtenir le modèle

### Option 1: Entraîner votre propre modèle ⭐ RECOMMANDÉ

Suivez le guide dans [`ENTRAINEMENT_RAPIDE.md`](../ENTRAINEMENT_RAPIDE.md):
- **Temps**: 15-20 minutes avec 10 images
- **Plateforme**: Google Colab (gratuit)
- **Résultat**: Modèle adapté à vos images

### Option 2: Utiliser un modèle pré-entraîné

Si disponible, téléchargez depuis:
- **Hugging Face**: [Votre lien ici]
- **Google Drive**: [Votre lien ici]

### Option 3: Upload via l'application Streamlit

L'application permet d'uploader le modèle directement via l'interface:
1. Entraînez votre modèle
2. Téléchargez `void_detection_best.h5` depuis Google Drive
3. Dans l'app Streamlit, utilisez la section "Upload du modèle"

## 📝 Placement du modèle

Une fois obtenu, placez le fichier ici:
```
models/
└── void_detection_best.h5
```

## 🔧 Hébergement alternatif (pour déploiement)

Pour déployer sur Streamlit Cloud, hébergez le modèle sur:

### Hugging Face (Recommandé)
```bash
# 1. Créer un compte sur huggingface.co
# 2. Créer un nouveau Model repository
# 3. Uploader void_detection_best.h5
# 4. Obtenir l'URL de téléchargement
```

### Google Drive
```bash
# 1. Uploader le modèle sur Google Drive
# 2. Clic droit → Partager → Obtenir le lien
# 3. Mettre en "Accès: Tous ceux qui ont le lien"
# 4. Récupérer l'ID du fichier dans l'URL
```

Consultez [`DEPLOIEMENT_STREAMLIT.md`](../DEPLOIEMENT_STREAMLIT.md) pour plus de détails.

## ⚙️ Spécifications du modèle

- **Architecture**: U-Net optimisé
- **Input**: Images 384x384 ou 512x512 en niveaux de gris
- **Output**: Segmentation 3 classes (soudure, voids, fond)
- **Taille**: ~150 MB
- **Format**: Keras HDF5 (.h5)
- **Précision**: Dice coefficient > 0.80 (avec 10 images) ou > 0.85 (avec 30+ images)

## 🚫 Ne pas commiter dans Git

Le fichier `.gitignore` est configuré pour exclure:
- `models/*.h5`
- `models/*.keras`
- `*.h5`

Ceci évite de surcharger le repository GitHub.
