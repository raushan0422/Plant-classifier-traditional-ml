import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

from utils.data_loader import load_labels, load_images
from utils.feature_extraction import extract_features

# 1. Load dataset
print("🔹 Loading CSV data...")
df = load_labels("data/train.csv")

# 2. Convert one-hot encoded targets to single label
print("🔹 Converting one-hot encoded targets to label column...")
df['label'] = df[['healthy', 'multiple_diseases', 'rust', 'scab']].idxmax(axis=1)

image_ids = df['image_id'].values
labels = df['label'].values

# 3. Load and preprocess images
print("🔹 Loading and resizing images...")
images = load_images("data/raw", image_ids)

# 4. Feature extraction
print("🔹 Extracting handcrafted features (color + texture)...")
features = extract_features(images)

# 5. Save features to CSV
os.makedirs("features", exist_ok=True)
feature_df = pd.DataFrame(features)
feature_df['label'] = labels
feature_df.to_csv("features/feature_data.csv", index=False)
print("✅ Features saved to features/feature_data.csv")

# 6. Train/test split
print("🔹 Splitting dataset...")
X = feature_df.drop(columns=['label'])
y = feature_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 7. Train classifiers
print("🔹 Training models...")
os.makedirs("models", exist_ok=True)

models = {
    'svm': SVC(probability=True),
    'random_forest': RandomForestClassifier(n_estimators=100),
    'gbm': GradientBoostingClassifier()
}

for name, model in models.items():
    print(f"\n📌 Training: {name.upper()}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    joblib.dump(model, f"models/model_{name}.pkl")
    print(f"✅ Saved {name} model to models/model_{name}.pkl")

print("\n🎉 All models trained and saved successfully!")
