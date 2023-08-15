from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import requests
import numpy as np
import os
from io import BytesIO
import urllib.request

#model link : https://drive.google.com/file/d/1KK7Uvt4lBt5mG86WiHcmuj9zHzQoF_-y/view?usp=sharing
model = load_model(r'D:\DogBreedClassification\efficientnetb4Model.h5')
labelPath = r'D:\DogBreedClassification\breeds.names'

file = open(labelPath, 'r')
labels = file.read().strip().split('\n')

def downloadImageFromURL(image_url):
    print(image_url)
    image_name = image_url.split("/")[-1]
    image_path = "D:\DogBreedClassification\eg\\" + image_name

    try:
        conn = urllib.request.urlopen(image_url)
    except urllib.error.HTTPError as e:
        print('HTTPError: {}'.format(e.code))
        return [False, "404"]
    except urllib.error.URLError as e:
        print('URLError: {}'.format(e.reason))
        return [False, "URL error"]
    else:
        urllib.request.urlretrieve(image_url, image_path)
        return [True, image_path]

# This Function is For Pre-Processing the Image For Feeding into Neural Network
def predict_from_image(img_url):
    petDownloaded = downloadImageFromURL(img_url)
    if petDownloaded[0] is False:
        return False
    img = image.load_img(petDownloaded[1], target_size=(380,380))

    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    print(img_tensor)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.

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

    # plt.imshow(img_tensor[0])
    # plt.axis('off')
    # plt.show()

    return [orig_class,value]


img_url = 'https://en.wikipedia.org/wiki/File:Afra_013.jpg'    # dog
predict_from_image(img_url)
