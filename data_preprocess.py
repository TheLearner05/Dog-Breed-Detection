import numpy as np 
import os
data1 = r"E:\Datasets\images.tar\images\images"
data2 = r"C:\Users\jayhe\Downloads\dogImages\dogImages\train"
path1 = r"C:\Users\jayhe\Downloads\images\Images\\"
path2 = r"C:\Users\jayhe\Downloads\dogImages\dogImages\train\\"
dataset = r'E:\Datasets\dogdata'
breeds = []
  

for i in os.listdir(dataset):
    print(len(os.listdir(dataset+'/'+i)))
    i=i.lower()

    breeds.append(i)
    breeds.sort()

gg = np.unique(breeds)

print((len(gg)))
