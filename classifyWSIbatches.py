# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:41:20 2019

@author: DeepBlue
"""
from keras.models import load_model
import os
import numpy as np
from keras.preprocessing import image
import pandas as pd
import glob
import shutil
import datetime


DESKTOP = 'C:\\Users\\DeepBlue\\Desktop\\'
IMGIN = DESKTOP + 'image_in\\'
IMGOUT = DESKTOP + 'image_out\\'
"""
 Using the trained model for classifying the 299x299px batches with the file containing
 the results under the predicted percentage
"""   
model_path = 'C:\\Users\\DeepBlue\\Documents\\Huy\\Code_model\\incept12111453.h5'
question_path = IMGOUT + 'tissue\\'
model = load_model(model_path)
size = 299
scale = 8
img_tensors = np.zeros((0, size, size, 3))
images_per_row = 4
margin = 5
thredhold = 0.5

list = os.listdir(question_path) # dir is your directory path
number_files = len(list)

for i, filename in enumerate (os.listdir(question_path)):
    
    path = question_path + os.sep + filename
    img = image.load_img(path, target_size=(size, size))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    img_tensors = np.append(img_tensors, img_tensor, axis=0).astype(np.float32)
    #print('the image ' + str(i) + ' was predicted')
    print('\r{} of {} images were predicted'.format(i+1, number_files), end='')

predictions = model.predict(img_tensors)

#save the predictions result in a file named "prediction_results.csv"
res = pd.DataFrame(predictions)
res.columns = ["prediction"]
#Savefile name
savepath = str(datetime.datetime.today())[5:16]
savepath = savepath.replace('-', '').replace(' ', '').replace(':', '')
savepath = DESKTOP + 'classify\\' + savepath + 'prediction_result.csv'
res.to_csv(savepath)
print('\rThe training is completed')





epithelium = 'C:\\Users\\DeepBlue\\Desktop\\classify\\e\\'
nonepi = 'C:\\Users\\DeepBlue\\Desktop\\classify\\ne\\'
    
file_names = glob.glob(question_path + '*.jpg')

for i, filename in enumerate(file_names):
   df = pd.read_csv(savepath)
   epith_label = df['prediction'][i]
   if epith_label < thredhold:       
       shutil.copy(filename, epithelium + filename.split('\\')[-1])
   elif epith_label >= thredhold:
       shutil.copy(filename, nonepi + filename.split('\\')[-1])

print ('The classifying process is completed')
