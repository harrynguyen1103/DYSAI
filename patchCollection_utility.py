# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:49:21 2020

@author: DeepBlue
"""

import os
import shutil
import glob
import random
import pandas as pd
import datetime

###
#remove patches of a specific case and renumber the index(.pkl) and patch(.jpg)
###
def _checkpatchlabel_consistency():
    indices = df.index.tolist()
    consistency = True
    
    if len(filenames) != len(indices):
        print('Incosistency detected! _file number')
        consistency = False
    
    for f, i in zip(filenames, indices):
        if f == i:
            continue
        else:
            print('Incosistency detected! {}__{}'.format(f, i))
            consistency = False
    
    if consistency:
        print('Patch and label consistency have been confirmed. ({} files)'.format(len(filepaths)))
        
    return consistency

def _reform_patchcollection():
    df_r = df.drop(df[df['Case'] == CASE].index.tolist())
    for i in range(len(df_r)): 
        newpatchname = str(i).zfill(6)
        oldpatchname = df_r[i:i+1].index[0]
        idxdict = {oldpatchname:newpatchname}
        df_r = df_r.rename(index=idxdict)
        shutil.copy(IMGDIR + oldpatchname + '.jpg', NEWIMGDIR + newpatchname + '.jpg')
    return df_r

def utility_1():
    #IMGDIRのpatch setに対し、特定のcaseのpatchを除去し、labelも書き換えたpatch setをNEWIMGDIRに作成
    if _checkpatchlabel_consistency():
        res = input('Case {} patches will be deleted. OK? (Type Yes).'.format(CASE))
        if res == 'Yes':
            df = _reform_patchcollection()
            
            labelfilename = 'Label' + str(datetime.datetime.now())[:10] + '.pkl'
            df.to_pickle(NEWIMGDIR + labelfilename)

def utility_2():
    ####IMGDIRのpatchをclass別にフォルダ分け
    traintest = ['train\\', 'test\\', 'exam\\']
    classfolder = ['nrm\\', 'lg\\', 'hg\\', 'scc\\']
    # testcase = ['075', '081', '093', '101', '112', '122', '134', '145', '156', '167'] #testdirに送るcase list
    # validcase = ['006', '013', '016', '022', '035', '047', '059', '071', '168', '179']#validdirに送るcase list
    testcase = ['004', '012', '015', '021', '034', '046', '057', '068', '072', '075', '081', '093', '101', '112', '122', '134', '145', '156', '167', '178', '189', '190', '198', '202','214'] #testdirに送るcase list
    validcase = ['006', '013', '016', '022', '035', '047', '059', '071', '074', '076', '082', '094', '103', '113', '124', '135', '146', '157', '168', '179', '180', '191', '199', '203','215']#
    # testcase = ['001', '002'] #testdirに送るcase list
    # validcase = ['003', '004']#validdirに送るcase list
    if _checkpatchlabel_consistency():
        for filename in filenames:
            cs = df.loc[filename, 'Case']
            epi = df.loc[filename, 'Epithelium']
            lo = df.loc[filename, 'LoGrade']
            hi = df.loc[filename, 'HiGrade']
            inv = df.loc[filename, 'Invasion']
            
            idx0 = 0 
            if cs in testcase:
                idx0 = 1 
            elif cs in validcase:
                idx0 = 2
            
            idx1 = 0
            if inv:
                idx1 = 3
            elif hi:
                idx1 = 2
            elif lo:
                idx1 = 1
            
            if epi:     
                src = IMGDIR + filename + '.jpg'
                dst = DESKTOP1 + traintest[idx0] + classfolder[idx1] + filename + '.jpg'
                shutil.copy(src, dst)
            else:
                print('Stromal patch. {}'.format(filename))

def utility_3():
    
#    for filepath in filepaths:
#        if random.random() > 0.9:
#            dst = DESKTOP + 'train\\epi\\' + os.path.basename(filepath)
#        else:
#            dst = DESKTOP + 'test\\epi\\' + os.path.basename(filepath)
#        shutil.copy(filepath, dst)
    
    stromal_filepaths = glob.glob(DESKTOP + 'stroma_patches\\*.jpg')
    for filepath in stromal_filepaths:
        if random.random() > 0.9:
            dst = DESKTOP1 + 'train\\strm\\' + os.path.basename(filepath) 
        else:
            dst = DESKTOP1 + 'test\\strm\\' + os.path.basename(filepath)
        shutil.copy(filepath, dst)
        
####Initialization
DESKTOP = 'D:\\78_cases\\sakamoto\\'
DESKTOP1 = 'D:\\78_cases\\ishida\\'
IMGDIR = DESKTOP + 'patches\\' #must contain patch jpgs and the label
#NEWIMGDIR = DESKTOP + 'image_out\\'

filepaths = glob.glob(IMGDIR + '*.jpg')
filenames = [os.path.basename(x).replace('.jpg', '') for x in filepaths]
#labelpath = glob.glob(IMGDIR + '*.pkl') #assume only one pkl in IMGDIR
df = pd.read_pickle(DESKTOP1 +'ishida_label.pkl')

####Utility 1####
####IMGDIRのpatch setに対し、特定のcaseのpatchを除去し、labelも書き換えたpatch setをNEWIMGDIRに作成
#utility_1()
#CASE = '093' #case to drop from patch collection

####Utility 2####
####IMGDIRのpatchをclass別にフォルダ分け
utility_2()     
        
####Utility 3####
####IMGDIRのpatchをstromaとepitheliumフォルダに分ける
#utility_3()           
        
        

