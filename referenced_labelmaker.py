# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:29:20 2019

@author: Rascal

"""
import glob
import sys, os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
            
def is_marked(src): #check if src contains a mark
    color = {'red':(245, 30, 30), 'green':(30, 245, 30), 'blue':(30, 30, 245)}
    trgt = [] #[R,G,B](boolean)
    threshold = 2 #minimum necessary dots regarded as marked
    for i, clr in enumerate(color):
        c1 = np.where(src[:, :, (0+i)%3] >= color[clr][(0+i)%3], 1, 0)
        c2 = np.where(src[:, :, (1+i)%3] <= color[clr][(1+i)%3], 1, 0)
        c3 = np.where(src[:, :, (2+i)%3] <= color[clr][(2+i)%3], 1, 0)
        trgt.append((np.count_nonzero(c1 * c2 * c3) >= threshold)) 
    return trgt #list of booleans [R,G,B]

def save_labelfile():
    labelfilename = 'RefLabelNew' + str(datetime.datetime.now())[:10] + '.pkl'
    df.to_pickle(DESKTOP + labelfilename)
    print('The label has been saved as {} on desktop.'.format(labelfilename))

def _patchlabel(number=0):
    trgt = df.loc[str(number).zfill(6)]
    return trgt

def check_consistency(folder1, folder2):
    path1 = glob.glob(folder1 + '*_i.jpg')
    path2 = glob.glob(folder2 + '*_i.jpg')
    flag = True
    if len(path1) != len(path2):
        print('Error: Filenumber mismatch')
        flag = False
    
    for p1, p2 in zip(path1, path2):
        if Image.open(p1).size != Image.open(p2).size:
            print('{} Error: Size mismatch'.format(os.path.basename(p1)))
            flag = False
        if np.sum(np.array(Image.open(p1).crop((0,0,5,5)))) != \
        np.sum(np.array(Image.open(p2).crop((0,0,5,5)))):
            
            print('{} Error: Upside down? Recheck.'.format(os.path.basename(p1)))
            flag = False
        print('>', end='')
        
    return flag

#folder1 = "C:\\Users\\DeepBlue\\Desktop\\i_sakamoto\\"
#folder2 = "C:\\Users\\DeepBlue\\Desktop\\huydata\\"
#check_consistency(folder1, folder2)
        

##############
#Usage
#1. Create a patch set by 'guided_patchcollector0312.py'
#2. Copy the label as 'reference_label.pkl' on DESKTOP
#3. Put annotated wsi (such as 000_i.jpg) in BASEFOLDER.
#   Note that the wsi set must be identical to the one used in #1.
#4. Run. Annotations are extracted and saved as RefLabelNew**.

Image.MAX_IMAGE_PIXELS = 5000000000
DESKTOP = 'C:\\Users\\DeepBlue\\Desktop\\'
PIXEL = 400 #image pixel taken from WSI
COLORDICT = {'red':'HiGrade', 'green':'LoGrade', 'blue':'Invasion'} #dict of mark colors
BASEFOLDER = DESKTOP + 'i_wsi\\' #wsi marked
ORIGINALFOLDER = DESKTOP + 'i_sakamoto\\'
reference_path = DESKTOP + 'reference_label.pkl'
COLUMNS = ['Case', 'CordX', 'CordY', 'Epithelium', 'LoGrade', 'HiGrade', 'Invasion',\
            'Inflammation', 'Keratinization']
ref = pd.read_pickle(reference_path)
df = ref.copy(deep=True)
df.loc[:,['LoGrade', 'HiGrade', 'Invasion', 'Inflammation', 'Keratinization']] = 0

if not check_consistency(BASEFOLDER, ORIGINALFOLDER):
    print('Error: Inconsistency detected')
    sys.exit()

cases = list(dict.fromkeys(ref.loc[:, 'Case'].values.tolist()))

for case in cases:
    img_i = Image.open(BASEFOLDER + case + '_i.jpg')  
    idxs = ref[ref['Case']==case].index.tolist()
    cords = ref[ref['Case']==case].loc[:,['CordX', 'CordY']].values.tolist()
    if len(idxs) != len(cords):
        print('Error: Inconsistency detected')
        sys.exit()

    for idx, cord in zip(idxs, cords):
        _img_i = img_i.crop((cord[1], cord[0], cord[1] + PIXEL, cord[0] + PIXEL))
        mark = is_marked(np.array(_img_i))

        for i, clr in enumerate(COLORDICT):
            df.loc[idx, COLORDICT[clr]] = int(mark[i])
    print(case)            
    print('Total: {}'.format(len(idxs)))
    _df = df.loc[:, ]
    for l in COLUMNS[4:7]:
        print('{}: {}'.format(l, (df[df.loc[:, 'Case'] ==case].loc[:, l].sum())))
        
save_labelfile()
