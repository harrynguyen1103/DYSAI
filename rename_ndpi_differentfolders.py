# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:42:15 2020

@author: Rascal
"""
import os
import glob
import shutil
import pandas as pd

def _check_consistency(path1, path2):
    flag = True
    for p1, p2 in zip(path1, path2):
        if p1 != p2.rstrip('.ndpa'):
            print('{} and {}'.format(p1, p2))
            print('Inconsistency detected')
            flag = False  
    return flag

########
#copy .ndpi and .ndpi.ndpa in source into destination with new names referring to case_renumberlist.csv
             
DESKTOP = 'C:\\Users\\DeepBlue\\Desktop\\'

source_ndpi = DESKTOP + 'wada\\'
source_ndpa = DESKTOP + 'wada\\'
destination = DESKTOP + 'destination\\'

df = pd.read_csv(DESKTOP + 'case_renumberlist.csv', header=None)


###ndpi and ndpa
ndpi_paths = sorted(glob.glob(source_ndpi + '*.ndpi'))
ndpa_paths = sorted(glob.glob(source_ndpa + '*.ndpi.ndpa'))

if not _check_consistency(ndpi_paths, ndpa_paths):
    print('Inconsistency detected')
else:
    ndpi_list = [int(os.path.basename(ndpi)[-13:-4].replace('.', '')) for ndpi in ndpi_paths]
    ndpa_dict = {int(os.path.basename(ndpa)[-18:-9].replace('.', '')):ndpa for ndpa in ndpa_paths}
    df_list = df.loc[:, 0].tolist()
    
    for i, original_name in enumerate(ndpi_list):
        if original_name in df_list:
            if not original_name in ndpa_dict.keys():
                print('No ndpa correponding to the ndpi {}'.format(original_name))
            else:
                
                old_ndpi = ndpi_paths[i]
                old_ndpa = ndpa_dict[original_name]
            
                new_ndpi = destination + str(df[df.loc[:, 0] == original_name][1].values[0]).zfill(3) + '.ndpi'
                new_ndpa = destination + str(df[df.loc[:, 0] == original_name][1].values[0]).zfill(3) + '.ndpi.ndpa'
                
#                print(i)
#                print(old_ndpi, old_ndpa)
#                print(new_ndpi, new_ndpa)
##    #           
#                os.rename(old_ndpi, new_ndpi)
#                os.rename(old_ndpa, new_ndpa)
#                
                shutil.copy(old_ndpi, new_ndpi)
                shutil.copy(old_ndpa, new_ndpa)

###only ndpi
#ndpi_paths = sorted(glob.glob(source_ndpi + '*.ndpi'))
#ndpi_list = [int(os.path.basename(ndpi)[-13:-4].replace('.', '')) for ndpi in ndpi_paths]
#df_list = df.loc[:, 0].tolist()
#    
#for i, original_name in enumerate(ndpi_list):
#    if not original_name in df_list:
#        print('{} no corresponding file in csv'.format(original_name))
#    old_ndpi = ndpi_paths[i]
#    new_ndpi = destination + str(df[df.loc[:, 0] == original_name][1].values[0]).zfill(3) + '.ndpi'
#
#    print(old_ndpi, new_ndpi)
          
#    os.rename(old_ndpi, new_ndpi)
