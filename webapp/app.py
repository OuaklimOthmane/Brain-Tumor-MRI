"""Flask app for brain tumor MRI classification."""
import os
import io
import numpy as np
import cv2
import joblib
from flask import Flask, render_template, request, jsonify

from features import preprocess, extract_features

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load artifacts ───────────────────────────────────────────────────────────
def _load_models():
    paths = {name: os.path.join(MODELS_DIR, f"{name}.joblib")
             for name in ("scaler", "knn", "rf", "meta")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Modèles manquants. Lance d'abord la section 14 du notebook pour les générer.\n"
            f"Fichiers manquants : {missing}")
    return (joblib.load(paths["scaler"]),
            joblib.load(paths["knn"]),
            joblib.load(paths["rf"]),
            joblib.load(paths["meta"]))


scaler, knn, rf, meta = _load_models()
CLASSES  = meta["classes"]
IMG_SIZE = meta["img_size"]
BEST_K   = meta["best_k"]

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024   # 8 MB


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, best_k=BEST_K)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image envoyée."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Fichier vide."}), 400

    # Read image from upload
    data = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Image invalide ou format non supporté."}), 400

    # Preprocess + extract features
    img       = preprocess(img_bgr, IMG_SIZE)
    feats     = extract_features(img).reshape(1, -1)
    feats_sc  = scaler.transform(feats)

    # Predict with both models
    def _predict(model, name):
        pred = int(model.predict(feats_sc)[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feats_sc)[0].tolist()
        else:
            proba = [1.0 if i == pred else 0.0 for i in range(len(CLASSES))]
        return {
            "model": name,
            "label": CLASSES[pred],
            "label_index": pred,
            "probabilities": [{"class": c, "p": round(float(p), 4)}
                              for c, p in zip(CLASSES, proba)],
            "confidence": round(float(max(proba)), 4),
        }

    return jsonify({
        "knn": _predict(knn, f"KNN (K={BEST_K})"),
        "rf":  _predict(rf,  "Random Forest"),
        "classes": CLASSES,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
