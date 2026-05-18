#!/bin/bash
# Double-cliquez sur ce fichier dans le Finder pour lancer NeuroScan.

# Aller dans le dossier du projet
cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════"
echo "  NeuroScan — Brain Tumor MRI Classifier"
echo "════════════════════════════════════════════════════"
echo ""

# Vérifier que les modèles existent
if [ ! -f "models/knn.joblib" ] || [ ! -f "models/rf.joblib" ] || [ ! -f "models/scaler.joblib" ]; then
    echo "❌ Modèles non trouvés dans ./models/"
    echo ""
    echo "   Lance d'abord la Section 14 du notebook brain_tumor.ipynb"
    echo "   pour générer scaler.joblib, knn.joblib, rf.joblib, meta.joblib."
    echo ""
    read -p "Appuie sur Entrée pour fermer cette fenêtre..."
    exit 1
fi

# Vérifier que le venv existe
if [ ! -f ".venv/bin/python" ]; then
    echo "❌ Environnement virtuel .venv introuvable."
    echo ""
    read -p "Appuie sur Entrée pour fermer cette fenêtre..."
    exit 1
fi

# Si le port 5000 est déjà occupé, prévenir
if lsof -i :5000 >/dev/null 2>&1; then
    echo "⚠️  Le port 5000 est déjà utilisé."
    echo "   Ouverture du navigateur sur l'app existante..."
    open "http://127.0.0.1:5000"
    echo ""
    read -p "Appuie sur Entrée pour fermer cette fenêtre..."
    exit 0
fi

echo "✅ Démarrage du serveur Flask..."
echo "   URL : http://127.0.0.1:5000"
echo ""
echo "   (Pour arrêter le serveur : Ctrl+C dans cette fenêtre)"
echo ""

# Ouvrir le navigateur après que Flask soit prêt (2s de marge)
(sleep 2 && open "http://127.0.0.1:5000") &

# Lancer Flask (bloque ici jusqu'à Ctrl+C)
.venv/bin/python webapp/app.py
