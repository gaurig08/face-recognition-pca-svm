"""
Utility functions for PCA + SVM Face Recognition
"""

from sklearn.decomposition import PCA
from sklearn.svm import SVC

def apply_pca(X, n_components=150):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca

def train_svm(X, y, C=100, gamma=0.0001):
    model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
    model.fit(X, y)
    return model
