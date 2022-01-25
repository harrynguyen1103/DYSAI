# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:39:33 2020

@author: Moon
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:29:20 2019

@author: DeepBlue

""" 
from PIL import Image
import shutil
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import pandas as pd
import datetime
import glob, os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plt
import itertools
import tkinter, tkinter.filedialog, tkinter.messagebox
import pickle
import time

Image.MAX_IMAGE_PIXELS = 5000000000
DESKTOP = 'C:\\Users\\Moon\\Desktop\\'
IMGIN = DESKTOP + 'image_in\\'
IMGOUT = DESKTOP + 'image_out\\'
PIXEL = 400
IMGSIZE = 299 
# IDLIST = ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009",
#           "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019",
#           "0020", "0021", "0022", "0023", "0024", "0025", "0026", "0027", "0028", "0029",
#           "0030", "0031", "0032", "0033", "0034", "0035", "0036", "0037", "0028", "0039",
#           "0040", "0041", "0042", "0043", "0044", "0045", "0046", "0047", "0048", "0049",
#           "0050", "0051", "0052", "0053", "0054", "0055", "0056", "0057", "0058", "0059",
#           "0060", "0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069",
#           "0070", "0071", "0072", "0073", "0074", "0075", "0076", "0077", "0078", "0079",
# "0080", "0081", "0082", "0083", "0084", "0085", "0086", "0087", "0088", "0089",
# "0090", "0091", "0092", "0093", "0094", "0095", "0096", "0097", "0098", "0099",
# "0100", "0101", "0102", "0103", "0104", "0105", "0106", "0107", "0108", "0109",
# "0110", "0111", "0112", "0113", "0114", "0115", "0116", "0117", "0118", "0119",
# "0120", "0121", "0122", "0123", "0124", "0125", "0126", "0127", "0128"]
stepsize = 400

def crop_WSI(n):
    
    imgfiles = glob.glob(IMGIN + str(n) + '.jpg')
    IMGCASE = IMGOUT + str(n) +'\\'
    
    if imgfiles == []:
        print('There is no file in folder image_in')
        
    else:
        cnt = 0
        number_WSI = 0
        for imgfile in imgfiles:
            print('\r{} of {} processing '.format(number_WSI+1, len(imgfiles)), end='')
            img = Image.open(imgfile)
            x = img.size[0]
            y = img.size[1]
            width_ns = x//PIXEL + 1
            height_ns = y//stepsize + 1
            number_WSI += 1
            for i in range(height_ns):
                top = PIXEL * i
                bottom = PIXEL * (i+1)
                for j in range(width_ns):
                    left = PIXEL + stepsize*j
                    right = left + PIXEL
                    img_crop = img.crop((left, top, right, bottom)).resize((IMGSIZE, IMGSIZE))
                    savefilename = IMGCASE + str(n) + "_" + str(cnt).zfill(5) + '.jpg'
                    img_crop.save(savefilename)
                    cnt += 1 #important to jumpt to next step
        print("\rThe cropping process of case number " + str(n) + " is completed")
 
def extract_blank(n):
    IMGCASE = IMGOUT + str(n) +'\\'
    tissue = IMGCASE + 'tissue\\'
    notissue = IMGCASE + 'notissue\\'
    threshold = 8000
    # threshold = 240 
    file_names = glob.glob(IMGCASE + '*.jpg')
    
    for filename in file_names:
        img = os.path.getsize(filename)
        # img = Image.open(filename) 
        # img = np.asarray(img)
        # img = Image.fromarray(img.astype(np.uint8)).resize((IMGSIZE, IMGSIZE))
        # img = Image.fromarray(img.astype(np.uint8))
        # img = np.array(img)
        if img < threshold:
           shutil.copy(filename, notissue + filename.split('\\')[-1])
        else:
           shutil.copy(filename, tissue + filename.split('\\')[-1])
    print ('The tissue-classifying process of case number ' + str(n) + ' is completed')

# def extract_blank_1(n):
#     IMGCASE = IMGOUT + str(n) +'\\'
#     tissue = IMGCASE + 'tissue\\'
#     notissue = IMGCASE + 'notissue\\'
#     threshold = 240
#     # threshold = 240 
#     file_names = glob.glob(IMGCASE + '*.jpg')
    
#     for filename in file_names:
#         img = Image.fromarray(filename.astype(np.uint8)).resize((IMGSIZE, IMGSIZE))
#         img = np.array(img)
#         if np.mean(img) > threshold:
#                 skiplist.append((j, i))
#             else:
#                 img_list.append(img)
#     img_npy = np.array(img_list)
#         img = os.path.getsize(filename)
#         # img = Image.open(filename) 
#         # img = np.asarray(img)
#         # img = Image.fromarray(img.astype(np.uint8)).resize((IMGSIZE, IMGSIZE))
#         # img = Image.fromarray(img.astype(np.uint8))
#         # img = np.array(img)
#         if img < threshold:
#            shutil.copy(filename, notissue + filename.split('\\')[-1])
#         else:
#            shutil.copy(filename, tissue + filename.split('\\')[-1])
#     print ('The tissue-classifying process of case number ' + str(n) + ' is completed')
       
def epi_predict(n):
    IMGCASE = IMGOUT + str(n) +'\\'
    tissue = IMGCASE + 'tissue\\'
    """
     Using the trained model for classifying the 299x299px batches with the file containing
     the results under the predicted percentage
    """   
    model_path = DESKTOP + 'epi_str_Incep.h5'
    question_path = tissue
    model = load_model(model_path)
    size = 299
    img_tensors = np.zeros((0, size, size, 3))
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
    # savepath = str(datetime.datetime.today())[5:16]
    # savepath = savepath.replace('-', '').replace(' ', '').replace(':', '')
    # savepath = DESKTOP  + savepath + 'prediction_result.pkl'
    savepath = DESKTOP + str(n) + '_epiprediction.pkl'
    res.to_csv(savepath)
    print('\rThe prediction process is completed')


    epi = IMGCASE + 'epi\\'
    nonepi = IMGCASE + 'nonepi\\'
        
    file_names = glob.glob(question_path + '*.jpg')
    
    for i, filename in enumerate(file_names):
       df = pd.read_csv(savepath)
       epith_label = df['prediction'][i]
       if epith_label < thredhold:       
           shutil.copy(filename, nonepi + filename.split('\\')[-1])
       elif epith_label >= thredhold:
           shutil.copy(filename, epi + filename.split('\\')[-1])
    
    print ('The epithelium-classifying process is completed')

    
""" predict patch tiles---------------------------------------------------------"""
def predict_patch(n):    
 
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
    
    """get prediction"""
    IMGCASE = IMGOUT + str(n) +'\\'
    epi = IMGCASE + 'epi\\'
    # testAI = DESKTOP + '203 cases\\128_25_50_training\\'
    img_size = 299
    COLUMNS=['Epi', 'Normal', 'LG', 'HG', 'SCC']
    testimage_dir = epi
    premodel_location = DESKTOP + 'epi_str_Incep.h5'
    
    model_location = DESKTOP + 'saka_sof_90_100.h5' 
    _factor = int(os.path.basename(model_location).replace('saka_sof_90_', '').replace('.h5', ''))/100
    # factor = os.path.basename(model_location).replace('saka_sof_20_', '').replace('.h5', '')
    
    premodel = load_model(premodel_location)
    model = load_model(model_location, custom_objects={'loss_function': loss_function})
    pred = analyze_imgfolder(premodel, model, testimage_dir)
    print ('The prediction process could spend much time for starting')
    pred.to_pickle(DESKTOP + str(n) + '_ratio_128_AI_90.pkl')
    print ('The prediction process of case number ' + str(n) + ' is completed')
    



"""------------------------------------"""
"""select thredholds for each class of each tile and binary them"""


def select_file(idir, filetype, mes1='Select file', mes2='Select the file'):
    root = tkinter.Tk()
    root.withdraw()
    filetype = [('', '*.' + filetype)]
    tkinter.messagebox.showinfo(mes1, mes2)
    file = tkinter.filedialog.askopenfilename(filetypes = filetype, initialdir = idir)
    return file

def select_directory(idir, mes1='Select folder', mes2='Select folder'):
    root = tkinter.Tk()
    root.withdraw()
    tkinter.messagebox.showinfo(mes1, mes2)
    folder = tkinter.filedialog.askdirectory(initialdir = idir)
    return folder


def _tobinary(src):
    idx = np.argmax(src, axis=1)
    src_binary = np.zeros(src.shape).astype(np.int32)
    for i, j in enumerate(idx):
        src_binary[i, j] = 1
    return src_binary
            


def convert_label(src, idx):
    param0 = (-1, -0.33, 0.33, 1) #equal
    param1 = (-1, 0.8, 0.9, 1) #norm or not
    param2 = (-1, -0.9, -0.8, 1)#cancer or not
    param3 = (-1, -0.9, 0.9, 1)#norm/LG or HG/SCC
    param4 = (-1, -0.05, 0.05, 1)#norm, dys, SCC
    params = (param0, param1, param2, param3, param4)
    
    param = np.array(params[idx])
    return np.dot(src, param)
  
"""---------------------------"""
def startprocess_message(msg):
    global start_time
    start_time = time.time()
    print(msg, end='..')

def endprocess_message():
    elapsed_time = time.time() - start_time
    print('Finished in {:.2f} sec'.format(elapsed_time))
"""Counting the percentage of each class for each case"""
    
##########Main
# IDLIST = ["0078", "0079"]
# IDLIST = ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019"]
# IDLIST = ["0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019"]
# IDLIST = ["0016", "0017", "0018", "0019","0020", "0021", "0022", "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031", "0032"]                                                   
# IDLIST = ["0027", "0028", "0029", "0030", "0031","0032", "0033", "0034", "0035", "0036", "0037", "0038", "0039", "0040", "0041", "0042", "0043", "0044", "0045"]
# IDLIST = ["0043", "0044", "0045","0046" , "0047", "0048", "0049"]
# IDLIST = ["0050", "0051", "0052", "0053", "0054", "0055", "0056", "0057", "0058", "0059", "0060", "0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069", "0070", "0071"] 
           
# IDLIST = ["0060", "0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069"]
# IDLIST = ["0059", "0060", "0061", "0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069", "0070", "0071"]
# IDLIST = ["0062", "0063", "0064", "0065", "0066", "0067", "0068", "0069", "0070", "0071","0072", "0073", "0074", "0075", "0076", "0077", "0078", "0079"]
# IDLIST = ["0072", "0073", "0074", "0075", "0076", "0077", "0078", "0079", "0080", "0081", "0082"]
# IDLIST = ["0083", "0084", "0085", "0086", "0087", "0088", "0089","0090", "0091", "0092", "0093", "0094", "0095"]
# IDLIST = ["0096", "0097", "0098", "0099","0100", "0101", "0102", "0103", "0104", "0105", "0106", "0107", "0108", "0109", "0110", "0111", "0112", "0113", "0114", "0115", "0116", "0117", "0118", "0119"]
# IDLIST = ["0112", "0113", "0114", "0115", "0116", "0117", "0118", "0119",
# "0120", "0121", "0122", "0123", "0124", "0125", "0126", "0127", "0128"]
IDLIST = ["0128"]

"""Image processing"""
for n in IDLIST:
    # startprocess_message('The test process of case number ' + str(n) + ' starts')
    # crop_WSI(n)                                                                                                                                                             
    # extract_blank(n)
    # epi_predict(n)
    predict_patch(n)
    # with open('C:\\Users\\Moon\\Desktop\\' + str(n) + '_set_93_AI_5.pkl', 'rb') as f:
    #     cases = pickle.load(f)
    #     cases = cases.loc[:, ['Normal', 'LG', 'HG', 'SCC']].values
    #     cases = _tobinary(cases)
    #     # # print(n)
        
    #     # print(cases)
    #     percenlist = ['Normal', 'LG', 'HG', 'SCC']
    #     for i in range(4):
    #         percenlist[i] = sum(cases[:,i])/len(cases)
    #         print(percenlist[i])
    #     if sum(cases[:, 3]) >= 1:
    #         print('Case number ' + str(n) + ' is diagnosed as SCC')
    #     elif sum(cases[:, 2]) >= 1:
    #         print('Case number ' + str(n) + ' is diagnosed as high grade dysplasia')
    #     elif sum(cases[:, 1]) >= 1:
    #         print('Case number ' + str(n) + ' is diagnosed as low grade dysplasia')
    #     else:
    #         print('Case number ' + str(n) + ' is diagnosed as normal')
    #     print(len(cases))
    #     endprocess_message()
    # print(' ')
    


def create_folders():
    for n in IDLIST:
        path_case = 'C:\\Users\\Moon\\Desktop\\image_out\\' + str(n)
        path_tissue = 'C:\\Users\\Moon\\Desktop\\image_out\\' + str(n) + '\\tissue'
        path_notissue = 'C:\\Users\\Moon\\Desktop\\image_out\\' + str(n) + '\\notissue'
        path_epi = 'C:\\Users\\Moon\\Desktop\\image_out\\' + str(n) + '\\epi'
        path_nonepi = 'C:\\Users\\Moon\\Desktop\\image_out\\' + str(n) + '\\nonepi'
    
        os.makedirs(path_case) 
        os.makedirs(path_tissue)
        os.makedirs(path_notissue)
        os.makedirs(path_epi)
        os.makedirs(path_nonepi)
        print('Successfully created the directory')
# create_folders()
































