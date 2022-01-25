# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:09:58 2020

@author: DeepBlue
"""
### models MUST have be trained with an identical set of 'exam' patches.
### model name should be a surname.pkl for human label and a model_surname.pkl
### for AI model
import sys, os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import itertools
import tkinter, tkinter.filedialog, tkinter.messagebox
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes, players=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')

#    print(cm)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    x_label = players[0][1] + ' ' + players[0][0]
    y_label = players[1][1] + ' ' + players[1][0]
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.savefig(desktop + x_label + '_vs_ ' + y_label + '.png')
#    plt.tight_layout()

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

def _check_consistency(src1, src2):
    return len(src1) == len(src2)

def _tobinary(src):
    idx = np.argmax(src, axis=1)
    src_binary = np.zeros(src.shape).astype(np.int32)
    for i, j in enumerate(idx):
        src_binary[i, j] = 1
    return src_binary
            
def evaluate_similarity(humans, models): #humans, models; dict
    platform_names = ['human', 'AI']
    player_names = []  
    
    for name in surnames:
        for p in platform_names:
            player_names.append((name, p))
    
    for c in itertools.combinations(player_names, 2):
        if c[0][1] == 'human':
            soln1 = humans[c[0][0]]
        else:
            soln1 = models[c[0][0]]
        
        if c[1][1] == 'human':
            soln2 = humans[c[1][0]]
        else:
            soln2 = models[c[1][0]]
        
        soln1 = soln1.loc[:, ['Normal', 'LG', 'HG', 'SCC']].values
        soln2 = soln2.loc[:, ['Normal', 'LG', 'HG', 'SCC']].values
##Euclid distance
        print('{} vs {}'.format(c[0], c[1]))
#        print('L2 norm: {:.2f}'.format(np.linalg.norm(soln1-soln2)))
        
##graded Euclid distance
        val1 = []
        _labels = ['Equal', 'Norm:Abnorm', 'Ca:no_Ca', 'Norm/LG:HG/SCC', 'Norm:Dys:SCC']
        for idx in range(5):      
            soln1_g = convert_label(soln1, idx)
            soln2_g = convert_label(soln2, idx)
            val1.append(np.linalg.norm(soln1_g - soln2_g))    
            print('{} {:.2f}'.format(_labels[idx], np.linalg.norm(soln1_g - soln2_g)))
        
        draw_radarchart(_labels, val1, 'radar')

##Cosine similarity:
#        s1 = soln1.flatten() - np.mean(soln1)
#        s2 = soln2.flatten() - np.mean(soln2)
#        cos_similarity = np.dot(s1, s2)/(np.linalg.norm(s1) * np.linalg.norm(s2))
#        print('cosine similarity: {:.2f}'.format(cos_similarity))

##Confusion matrix
        cm = confusion_matrix(np.argmax(soln1, axis=1), np.argmax(soln2, axis=1),\
                              labels=[0, 1, 2, 3])
        np.set_printoptions(precision=2)
        
        plot_confusion_matrix(cm, classes=['Normal', 'LG', 'HG', 'SCC'],
                              players=c, normalize=True,
                              title='Confusion matrix')
#        plt.show()
#        plt.close()
##Accuracy        
        if c[0][1] == 'AI':
            soln1 = _tobinary(soln1)
        if c[1][1] == 'AI':
            soln2 = _tobinary(soln2)
        judge = np.all((soln1==soln2)==True, axis=1)
        concordance = np.sum(judge)/len(judge)
        print('concordance: {:.2f} \r\n'.format(concordance))     


def draw_radarchart(labels, values, imgname):
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    values = np.concatenate((values, [values[0]]))  # 閉じた多角形にする
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, 'o-')  # 外枠
    ax.fill(angles, values, alpha=0.25)  # 塗りつぶし
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)  # 軸ラベル
    ax.set_rlim([0,25])
#    fig.savefig(imgname)
#    plt.close(fig)

def convert_label(src, idx):
    param0 = (-1, -0.33, 0.33, 1) #equal
    param1 = (-1, 0.8, 0.9, 1) #norm or not
    param2 = (-1, -0.9, -0.8, 1)#cancer or not
    param3 = (-1, -0.9, 0.9, 1)#norm/LG or HG/SCC
    param4 = (-1, -0.05, 0.05, 1)#norm, dys, SCC
    params = (param0, param1, param2, param3, param4)
    
    param = np.array(params[idx])
    return np.dot(src, param)
   
######### Main
        
desktop = 'C:\\Users\\DeepBlue\\Desktop\\Results_20200608\\compare\\'
surnames = ['huy', 'sakamoto']

humans = {}
models = {}

for name in surnames:
    humans[name] = pd.read_pickle(desktop + 'human_' + name + '_sof_100.pkl')
    models[name] = pd.read_pickle(desktop + 'AI_' + name + '_sof_100.pkl')

if len(surnames) > 1:
    for i in itertools.combinations(surnames, 2):
        if not _check_consistency(humans[i[0]], models[i[1]]):
            print('Inconsistency detected. {}'.format(i))
            sys.exit()
           
    evaluate_similarity(humans, models)
