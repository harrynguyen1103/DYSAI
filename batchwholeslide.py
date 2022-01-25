# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:41:42 2019

@author: DeepBlue
"""
    
from PIL import Image
import glob
import os
import shutil

Image.MAX_IMAGE_PIXELS = 5000000000
DESKTOP = 'C:\\Users\\DeepBlue\\Desktop\\'
IMGIN = DESKTOP + 'image_in\\'
IMGOUT = DESKTOP + 'image_out\\'
PIXEL = 1000
IMGSIZE = 299
IDNAME = 1

#Crop the whole slide image into 1000x1000px batches
def crop_WSI():
    imgfiles = glob.glob(IMGIN + '*.jpg')
    
    if imgfiles == []:
        print('There is no file in folder image_in')
        
    else:
        cnt = 0
        for imgfile in imgfiles:
        
            print('\r{} of {} processing '.format(cnt, len(imgfiles)), end='')
            img = Image.open(imgfile)
            x = img.size[0]
            y = img.size[1]
            width_ns = x//PIXEL + 1
            height_ns = y//PIXEL + 1
            for i in range(height_ns):
                top = PIXEL * i
                bottom = PIXEL * (i+1)
                for j in range(width_ns):
                    left = PIXEL * j
                    right = PIXEL * (j+1)
                    img_crop = img.crop((left, top, right, bottom)).resize((IMGSIZE, IMGSIZE))
                    savefilename = IMGOUT + str(IDNAME) + "-" + str(cnt).zfill(4) + '.jpg'
                    img_crop.save(savefilename)
                    cnt += 1 #important to jumpt to next step
        print("The process is completed")
 
def extract_blank():
    basedir = 'C:\\Users\\DeepBlue\\Desktop\\image_out\\'
    nontissue = 'C:\\Users\\DeepBlue\\Desktop\\image_out\\non\\'
    tissue = 'C:\\Users\\DeepBlue\\Desktop\\image_out\\tissue\\'
    thredhold = 8000
    file_names = glob.glob(basedir + '*.jpg')
    
    for filename in file_names:
        img = os.path.getsize(filename)
        if img < thredhold:
           shutil.copy(filename, nontissue + filename.split('\\')[-1])
        elif img >= thredhold:
           shutil.copy(filename, tissue + filename.split('\\')[-1])
    print ('The classifying process is completed')
       
if __name__ == '__main__':
    crop_image = crop_WSI()
    extract_blankimage = extract_blank()
    
    




