import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from numpy.linalg import norm
import tensorflow 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors 


feature_list=np.array(pickle.load(open('embeddings.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model=tensorflow.keras.Sequential([model, GlobalMaxPooling2D()
])

st.title("Fashion Recommendation System")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0    

def feature_extraction(img_path,model):  
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result=result / norm(result)

    return normalized_result
  
def recommend(features,feature_list):
    neighbours=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbours.fit(feature_list)

    distances,indices=neighbours.kneighbors([features])

    return indices

uploaded_file=st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image=Image.open(uploaded_file)
        st.image(display_image,width=200)

        features=feature_extraction(os.path.join('uploads',uploaded_file.name),model)
        #st.text(features)

        indices = recommend(features, feature_list)

# Create a row for the first two images
        col1, col2 = st.columns(2)

        with col1:
            st.image(filenames[indices[0][0]], width=200)

        with col2:
            st.image(filenames[indices[0][1]], width=200)

        # Create another row for the next two images
        col1, col2 = st.columns(2)

        with col1:
            st.image(filenames[indices[0][2]], width=200)

        with col2:
            st.image(filenames[indices[0][3]], width=200)

        # Create a final row for the last image
        col1, col2 = st.columns([1, 2])

        with col2:
            st.image(filenames[indices[0][4]], width=200)
                        
    else:
        st.header("Some error has occured")  


