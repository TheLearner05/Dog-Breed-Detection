from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import streamlit as st
from PIL import Image, ImageOps


# model link : https://drive.google.com/file/d/1KK7Uvt4lBt5mG86WiHcmuj9zHzQoF_-y/view?usp=sharing
model = load_model(r'D:\DogBreedClassification\efficientnetb4Model.h5')
labelPath = r'D:\DogBreedClassification\breeds.names'
# This Function is For Pre-Processing the Image For Feeding into Neural Network

file = open(labelPath, 'r')
labels = file.read().strip().split('\n')


def predict_from_image(img_path):
    
    size = (380,380)    
    img = ImageOps.fit(img_path, size, Image.ANTIALIAS)
    #img = image.load_img(img_path, target_size=(380, 380))
    # (height, width, channels)
    img_tensor = np.asarray(img)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor1 = img_tensor/255.

    pred = model.predict(img_tensor)

    # print(pred)
    # print(np.argmax(pred))
    predicted_class = labels[np.argmax(pred)]
    orig_class = predicted_class.replace(
        predicted_class, predicted_class[10:])
    value = max(pred)[labels.index(predicted_class)]
    
    print(orig_class)

    
    print(value)
    #print(f"{orig_class}\n{max(pred)[labels.index(predicted_class)]}")

    # print(selected_breed_list[predicted_class])
    #plt.imshow(img_tensor1[0])
    #plt.axis('off')
    #plt.show()
    return orig_class,value


st.write('''
        # Dog Breed Classification WebApp

        ''')
st.write("This is a Dog Breed Classification WebApp to classify dog in different 120 available classes")
img = st.file_uploader("please upload dog picture",type=['jpg','jpeg','png'])
img_path = r'D:\DogBreedClassification\OIP.jpg'
if img is None:
    print("please upload the picture correctly")
else:
    image = Image.open(img)
    st.image(image,use_column_width=True)
    breed,prob = predict_from_image(image)
    st.write(" It is ",breed)
    st.write("Probability: ",prob)

