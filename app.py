import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from utils.feature_extraction import extract_color_histogram, extract_texture_features

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

# ----------------------------
# Header
# ----------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #2E7D32;
        }
        .subtext {
            font-size: 1.1em;
            color: #444;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            margin-top: 20px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Upload a leaf image and choose a traditional ML model to predict the disease using handcrafted features (color + texture).</p>', unsafe_allow_html=True)

# ----------------------------
# Sidebar - Model selection
# ----------------------------
st.sidebar.title("🔧 Model Settings")
model_choice = st.sidebar.selectbox("Choose a model", ["SVM", "Random Forest", "Gradient Boosting"])

model_paths = {
    "SVM": "models/model_svm.pkl",
    "Random Forest": "models/model_random_forest.pkl",
    "Gradient Boosting": "models/model_gbm.pkl"
}

st.sidebar.markdown("📎 **Model Info**")
if model_choice == "SVM":
    st.sidebar.info("Support Vector Machine: High-dimensional decision boundary.")
elif model_choice == "Random Forest":
    st.sidebar.info("Random Forest: Ensemble of decision trees, robust to overfitting.")
else:
    st.sidebar.info("Gradient Boosting: Sequential learning, strong predictive power.")

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (128, 128))

    st.subheader("🖼️ Uploaded Image")
    st.image(image_rgb, width=300, caption="Input Image")

    try:
        # ----------------------------
        # Feature Extraction
        # ----------------------------
        st.subheader("🔍 Extracting Features...")
        color_feat = extract_color_histogram(image_resized)
        texture_feat = extract_texture_features(image_resized)
        combined_feat = np.concatenate([color_feat, texture_feat]).reshape(1, -1)

        # ----------------------------
        # Load selected model
        # ----------------------------
        model = joblib.load(model_paths[model_choice])
        pred = model.predict(combined_feat)[0]
        probs = model.predict_proba(combined_feat)[0]
        class_labels = model.classes_

        # ----------------------------
        # Show Prediction
        # ----------------------------
        st.success(f"✅ Predicted Class: **{pred}**")

        # ----------------------------
        # Plot Probabilities
        # ----------------------------
        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({"Class": class_labels, "Probability": probs})
        prob_df = prob_df.sort_values("Probability", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(prob_df["Class"], prob_df["Probability"], color="#66bb6a")
        ax.set_xlabel("Probability")
        ax.set_xlim([0, 1])
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")

else:
    st.warning("👈 Please upload an image to begin.")

# ----------------------------
# Footer
# ----------------------------
st.markdown('<div class="footer">🧠 Built using handcrafted features and traditional ML (SVM, RF, GBM)</div>', unsafe_allow_html=True)
