# Fashion Recommendation System

This is a fashion recommendation system built with Python, TensorFlow, and Streamlit. The system uses a pre-trained ResNet50 model to extract features from images and then recommends similar items based on these features.

## Dependencies

- Python
- TensorFlow
- Streamlit
- PIL (Python Imaging Library)
- NumPy
- scikit-learn
- pickle
- os

## How it works

1. **Feature Extraction**: The system uses a pre-trained ResNet50 model to extract features from images. The model is trained on the ImageNet dataset and is used without the top layer (include_top=False). The output of the model is then passed through a GlobalMaxPooling2D layer.

2. **Saving Uploaded Files**: The system provides a function to save uploaded files to a specified directory.

3. **Recommendation**: The system uses a Nearest Neighbors model from scikit-learn to find the most similar items based on the extracted features. The model uses the Euclidean distance metric and a brute force algorithm to find the nearest neighbors.

4. **User Interface**: The system uses Streamlit to create an interactive user interface. Users can upload an image, and the system will display the most similar items.

## Usage:
1] Git clone repository.

2] Open your terminal run streamlit run main.py.
