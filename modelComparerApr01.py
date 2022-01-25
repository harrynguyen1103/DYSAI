# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 11:13:48 2020

@author: DeepBlue
"""
import os
import sys
import warnings
warnings.simplefilter('ignore', FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #2:INFOとWARNINGが出ない。 0:すべて表示


import glob

import numpy as np

from PIL import Image
# from utilitiesRev import ImagePatch
# from utilitiesRev import make_dysplasiaLabel

# import matplotlib.pyplot as plt
# from matplotlib.path import Path
# from matplotlib.spines import Spine
# from matplotlib.projections.polar import PolarAxes
# from matplotlib.projections import register_projection

# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import to_categorical

from sklearn.metrics import mean_squared_error

def calculate_loss(ground_truth, predictions, weight = [0, 1, 2, 3]):
    truth = np.sum(ground_truth * weight, axis = 1)
    pred = np.sum(predictions * weight, axis = 1)
    return mean_squared_error(truth, pred)


def draw_radarchart(result_dict):
        #format data to fit radar chart routine
    _dict = ['R&C', 'R&B','C&C', 'C&B']

    model_names = list(result_dict[list(result_dict.keys())[0]])
    case_nums = list(result_dict.keys())
    
    res_box = []
    for case_num in case_nums:#str
        res = [result_dict[case_num][model_name] for model_name in model_names]
        res = np.concatenate((np.array(res), [np.array(res)[0]]))
        
        res_box.append(res)
        
    angles = np.linspace(0, 2 * np.pi, len(model_names) + 1, endpoint=True)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    
    for values in res_box:
        ax.plot(angles, values)  # 外枠
        # ax.fill(angles, values, alpha=0.25)  # 塗りつぶし
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, _dict)  # 軸ラベル
    ax.set_rlim(0 ,2.5)
    # ax.set_rgrid([0, 0.5, 1.0, 1.5, 2.0, 2.5])

    plt.savefig(DESKTOP + 'result.png', bbox_inches='tight')


# #############MAIN########
#Initialization
Image.MAX_IMAGE_PIXELS = 5000000000
DESKTOP = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop\\"
savefolder = DESKTOP + "image_out\\"

HUMAN_ANOTATION_VISUALIZATION = False #predictの代わりにhuman anotation (_iファイル)を使って描画する時にTrueにする
DISPLAY = 'gradation' #'class' or 'gradation'

XSIZE = 60
YSIZE = 1000
SKIPRATIO = 100
# MODELS = ['mobilenet1.h5', 'mobilenet2.h5', 'mobilenet3.h5', 'mobilenet4.h5']
MODELS = ['vgg1.h5', 'vgg2.h5', 'vgg3.h5', 'vgg4.h5']
sys.exit()
# img_folder = DESKTOP + "TongueBiopsySet1\\"

img_folder = DESKTOP + "biopsyTest\\"
model_folder = DESKTOP
img_guidepaths = glob.glob(img_folder + '*_t.jpg')
img_paths = [i.replace("_t", "_o") for i in img_guidepaths]

weight = [0, 1, 2, 3]
# weight = [0, 1, 8, 8]

tissue_list = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1, 2, 1, 2, 2, 
2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2,
2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
3, 4, 2, 1, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]

#get human annotation

result_dict = {}   
for img_path, img_guidepath in zip(img_paths, img_guidepaths):
    wsi_name = os.path.basename(img_path).rstrip("_o.jpg")
    imagepatch = ImagePatch(img_path=img_path, img_inkedpath=img_path.replace('_o', '_i'),
                            img_guidepath=img_guidepath, width=XSIZE, low=20,
                            high=240, skip_ratio=SKIPRATIO, min_distance=None,
                            savefolder=savefolder, wsi_name=wsi_name,
                            testmode=True)
    
    df, patchset = imagepatch.collect_patch()
    df = make_dysplasiaLabel(df)
    
    imgn = np.zeros((YSIZE, XSIZE, 3)).astype(np.uint8)
    
    patchn_list = []
    for patch in patchset:

         #clear
        imgn = np.zeros((YSIZE, XSIZE, 3)).astype(np.uint8)
        if patch.size[1] <= YSIZE:
            imgn[-patch.size[1]:, :, :] = np.array(patch) #基底側につめる
        else:
            imgn = np.array(patch)[-YSIZE:, :, :]
 
        patchn_list.append(imgn)
        
    patchsetn = (np.array(patchn_list) / 255).astype(np.float32)
    
    ground_truth = to_categorical(np.array(df['label']), 4)
    
    loss_dict = {}
    for modelname in MODELS:
        _modelname = modelname.replace('.h5', '')
        model = load_model(model_folder + modelname)
    
        predictions = model.predict(patchsetn)
        
        loss = calculate_loss(ground_truth, predictions)
        print('{} loss {}'.format(_modelname, loss))
        loss_dict[_modelname] = loss
    
    result_dict[wsi_name] = loss_dict

draw_radarchart(result_dict)
    

    

##########################
DESKTOP = 'C:\\Users\\Moon\\Desktop\\'
TRAINING = DESKTOP + "203_cases_800x_selected\\"
# ns = ["89_50_50_set_0"]
# ns = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# ns = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
ns = [0]
for n in ns:
    human = pd.read_pickle(TRAINING + 'human_103_50_50_set10x800sel_' + str(n) + '.pkl')
    model = pd.read_pickle(TRAINING + 'AI_103_50_50_set10x800sel_' + str(n) + '.pkl')

    #convert dataframe to numpy
    human = np.array(human)
    model = np.array(model)
    
    #delete "Epi" Column in numpy array "human" and "model"
    human1 = np.delete(human, 1, 1)
    model1 = np.delete(model, 1, 1)
    print('calculate_loss of' + str(n) + 'is: ')
    h = calculate_loss(human1, model1)
    print(h)
        
human1 = np.array([[1, 0, 0, 0],[1, 0, 0, 0]])

model1 = np.array([[0.5, 0.2, 0.3, 0], [0.5, 0.3, 0.2, 0]])    


#######################
def calculate_score(ground_truth, predictions, weight = [0, 1, 2, 3]):
    truth = np.sum(ground_truth * weight, axis = 1)
    pred = np.sum(predictions * weight, axis = 1)
    return truth, pred

## Import the packages
import numpy as np
from scipy import stats
import pandas as pd


###paired t-test for significant test human and AI's weighted scores
#create the arrays of weighted scores for human (human2) and AI (model2)
human2, model2 = calculate_score(human1, model1)
stats.ttest_rel(human2, model2)



##### test for significant between root mean squared error of two models
import pandas as pd
DESKTOP = 'C:\\Users\\Moon\\Desktop\\New folder\\'
human1 = pd.read_pickle(DESKTOP + 'human_sakamoto_sof_100.pkl')
model1 = pd.read_pickle(DESKTOP + 'AI_sakamoto_sof_100.pkl')
human2 = pd.read_pickle(DESKTOP + 'human_40_.pkl')
model2 = pd.read_pickle(DESKTOP + 'AI_40_.pkl')

human1 = np.array(human1)
human2 = np.array(human2)
model1 = np.array(model1)
model2 = np.array(model2)

human11 = np.delete(human1, 1, 1)
human22 = np.delete(human2, 1, 1)
model11 = np.delete(model1, 1, 1)
model22 = np.delete(model2, 1, 1)

loss1 = human11 - model11
loss2 = human22 - model22

N = len(loss1)
var_a = loss1.var(ddof=1)
var_b = loss2.var(ddof=1)
s = np.sqrt((var_a + var_b)/2)
t = (loss1.mean() - loss2.mean())/(s*np.sqrt(2/N))
df = 2*N - 2
p = 1 - stats.t.cdf(t,df=df)
print("t = " + str(t))
print("p = " + str(2*p))
t2, p2 = stats.ttest_ind(loss1,loss2)
print("t = " + str(t2))
print("p = " + str(p2))
    
### calculating mean absolute error
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_mae(ground_truth, predictions, weight = [0, 1, 2, 3]):
    truth = np.sum(ground_truth * weight, axis = 1)
    pred = np.sum(predictions * weight, axis = 1)
    return mean_absolute_error(truth, pred)


DESKTOP = 'C:\\Users\\Moon\\Desktop\\New folder\\'
# DESKTOP = 'C:\\Users\\Moon\\Desktop\\New folder\\103\\'

# ns = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# ns = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
ns = ['10_2', '30_6', '50_10', '70_14', '100_10', '100_20', '100_30', '100_40', '100_50']

for n in ns:
    human1 = pd.read_pickle(DESKTOP + 'human_' + n + '_50_soft_100.pkl')
    model1 = pd.read_pickle(DESKTOP + 'AI_' + n + '_50_soft_100.pkl')
    
    human1 = np.array(human1)
    model1 = np.array(model1)

    human11 = np.delete(human1, 1, 1)
    model11 = np.delete(model1, 1, 1)

    # print('MAE of AI ' + str(n) + ' is: ')
    print('MAE of AI ' + n + ' is: ')
    print(calculate_mae(human11,model11))



##### calculate the T-test for the means (MAE) of two independent samples of scores
import pandas as pd
import numpy as np
from scipy import stats

DESKTOP = 'C:\\Users\\Moon\\Desktop\\New folder\\'
DESKTOP1 = 'C:\\Users\\Moon\\Desktop\\New folder\\128\\'

# ns = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
ns = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


human1 = pd.read_pickle(DESKTOP + 'human_sakamoto_sof_100.pkl') #AI1-1
model1 = pd.read_pickle(DESKTOP + 'AI_sakamoto_sof_100.pkl')

weight = [0, 1, 2, 3]

for n in ns:
    human2 = pd.read_pickle(DESKTOP1 + 'human_' + str(n) +'_.pkl') #AI2
    model2 = pd.read_pickle(DESKTOP1 + 'AI_' + str(n) + '_.pkl')
    # human2 = pd.read_pickle(DESKTOP1 + 'human_103_50_50_set10x_'+ str(n) + '.pkl') #AI2
    # model2 = pd.read_pickle(DESKTOP1 + 'AI_103_50_50_set10x_' + str(n) + '.pkl')
    

    human1 = np.array(human1)
    human2 = np.array(human2)
    model1 = np.array(model1)
    model2 = np.array(model2)
    
    human11 = np.delete(human1, 1, 1)
    human22 = np.delete(human2, 1, 1)
    model11 = np.delete(model1, 1, 1)
    model22 = np.delete(model2, 1, 1)
    
    
    score1 = np.sum(human11 * weight, axis = 1) - np.sum(model11 * weight, axis = 1)
    score2 = np.sum(human22 * weight, axis = 1) - np.sum(model22 * weight, axis = 1)
    # score2 = human22 * weight - model22 * weight 
    # loss1 = human11 - model11
    # loss2 = human22 - model22

    print('significance test of DysAI1.1 and DysAI2 ' + str(n) + ' is:' )
    print(' ')
    print(stats.ttest_ind(np.abs(score1), np.abs(score2)))
    
    print('MAE of AI2 is')
    print(np.sum(np.abs(score2))/len(human22))

# np.sum(score1)/len(human11)













