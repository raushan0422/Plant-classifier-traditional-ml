import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_features(images):
    features = []
    for img in images:
        color_hist = extract_color_histogram(img)
        texture = extract_texture_features(img)
        combined = np.concatenate([color_hist, texture])
        features.append(combined)
    return np.array(features)
