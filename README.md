# Face Recognition System using PCA and SVM

Face recognition system built using classical machine learning techniques.
Dimensionality reduction is performed using PCA (Eigenfaces),
followed by classification using an RBF-kernel SVM.

## Dataset
- Labeled Faces in the Wild (LFW)
- Loaded directly using `sklearn.datasets.fetch_lfw_people`
- 7 classes, 1288 images

## Methodology
- Data standardization
- PCA for dimensionality reduction (1850 → 150 features)
- Explained variance retained: ~94.5%
- SVM (RBF kernel) with hyperparameter tuning using GridSearchCV

## Results
- Test Accuracy: ~85%
- Cross-validation accuracy: ~83%
- F1-score: 0.85

## Project Structure
- notebooks/: Full analysis, visualization, and experiments (Google Colab)
- src/: Reusable utility functions

## Future Improvements
- Try deep learning (CNN-based models)
- Increase dataset size
- Real-time face recognition pipeline
