# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 2020
@author: Rascal
"""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from keras.models import load_model
import tensorflow as tf
import keras.backend as K

def analyze_imgfolder(premodel, model, folderpath):
    #input: model, folder path to images
    #return: dataframe of prediction (Normal, LoGrade, HiGrade, SCC)
    
    _df = pd.DataFrame(index=[], columns=COLUMNS) #different from label df  
    filenames = glob.glob(folderpath + '*.jpg')
    for filename in filenames:
        img = Image.open(filename).resize((img_size, img_size))
        var = np.array(img) / 255.
        var = np.expand_dims(var, axis=0)
        
        epi_strm = premodel.predict(var)
        nrm_dys = model.predict(var)
         
        #the train used sigmoid instead of softmax.
        nrm_dys /= np.sum(nrm_dys) #sum of prediction = 1.
        sample_name = os.path.basename(filename).replace('.jpg','')
        
        _f = np.concatenate([epi_strm, nrm_dys], 1)
        _df.loc[sample_name] = _f[0]
        
    return _df  

def test_model(premodel, model, testimage_dir):
    ## prediction   
    preds = analyze_imgfolder(premodel, model, testimage_dir) #preds: dataframe
    
    ## label (multiclass label to single class label)
    labelpath =  glob.glob(testimage_dir + '*.pkl')    
    label = pd.read_pickle(labelpath[0])
    label['Epi'] = label['Epithelium']
    label['SCC'] = label['Invasion'] #if Invasion, SCC
    label['HG'] = label['HiGrade'] * (1 - label['SCC']) #elif HiGrade, HG
    label['LG'] = label['LoGrade'] * (1 - label['SCC']) * (1 - label['HG']) #elif LoGrade, LG
    label['Normal'] = (1 - label['SCC']) * (1 - label['HG']) * (1 - label['LG']) #else Normal   
      
    return label.loc[preds.index, COLUMNS].astype(int), preds

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def _prediction_toanswer(pred):  #disused
    #set zero-filled _label
    _label = pd.DataFrame(0, index = pred.index, columns=COLUMNS, dtype='int32')
  
    _idx = list(pred.index)
    _key = pred.idxmax('columns').values.tolist()
    
    #mark the answers
    for i, k in zip(_idx, _key):
        _label.loc[i, k] = 1
    
    return _label
    
def loss_function(y_true, y_pred, axis=-1):
    #allow low/high error, high/SCC error to some extent
#    modification_tensor = np.array([[1, 0, 0, 0], [0, 0.9, 0.1, 0],\
#                                    [0, 0.1, 0.9, 0], [0, 0, 0, 1]])
##   default setting: no modification
#    modification_tensor = np.diag([1, 1, 1, 1])

#    modification calculated from model name  
    modification_tensor = np.array([[1, 0, 0, 0], [0, _factor, 1 - _factor, 0],\
                                    [0, 1 - _factor, _factor, 0], [0, 0, 0, 1]])
    
    x = tf.constant(modification_tensor, dtype='float32')
    
    y_pred /= tf.reduce_sum(y_pred, axis, True)
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    
    y_true = K.dot(y_true, x) #multiply by modification tensor
           
    return - tf.reduce_sum(y_true * tf.log(y_pred), axis)

def ROC_curve(label, pred, key='Normal', overwrite=True):
    y_true = label.loc[:, key].values.tolist()
    y_score = pred.loc[:, key].values.tolist()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.plot(fpr, tpr, marker='o', label=key)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()
    drname = 'saka'
    stage = 'sof'
    if not overwrite:
        plt.text(0.7, 0.1, 'AUC={:.3f}'.format(auc), size=15)
        plt.savefig(desktop + 'ROC_' + key + '_' + drname + '_'+ stage + '_'+ factor + '.png')
    if overwrite:
        plt.legend()    
        plt.savefig(desktop + 'ROCs_all_'  + drname + '_'+ stage + '_'+ factor + '.png')
    if not overwrite:
        plt.close()
    
##########Main
img_size = 299
COLUMNS=['Epi', 'Normal', 'LG', 'HG', 'SCC']


"""check before running"""
desktop = 'C:\\Users\\Moon\\Desktop\\'
testimage_dir = desktop + 'exam\\'
premodel_location = desktop + 'epi_str_Incep.h5'
model_location = desktop + 'saka_sof_100.h5' #change this one
_factor = int(os.path.basename(model_location).replace('saka_sof_', '').replace('.h5', ''))/100
factor = os.path.basename(model_location).replace('saka_sof_', '').replace('.h5', '')

premodel = load_model(premodel_location)
model = load_model(model_location, custom_objects={'loss_function': loss_function})
label, pred = test_model(premodel, model, testimage_dir)

for c in COLUMNS[1:]:
    ROC_curve(label, pred, key=c, overwrite=False)
print("separated plots done")

for c in COLUMNS[1:]:
    ROC_curve(label, pred, key=c, overwrite=True)
print("overwrite plot done")

#change doctor's name corresponding to his/her model
# """check before running"""
drname = 'saka'
stage = 'sof'
label.to_pickle(desktop + 'human_' + drname + '_'+ stage + '_'+ factor + '.pkl')
pred.to_pickle(desktop + 'AI_' + drname + '_'+ stage + '_'+ factor + '.pkl')
