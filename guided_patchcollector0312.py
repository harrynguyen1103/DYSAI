# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:29:20 2019

@author: DeepBlue

"""
import sys, os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import random
import datetime



class ImagePatch():

    def __init__(self, imgpath, img_inkedpath):
        self.img_patches = [] #list of image
        self.img_location = [] #list of image location coordinates; pixels
        self.patch_list = [] #list of (x,y) coordinates; grids of patches
        self.mark_list = [] #list of boolean; [[R,G,B], [R,G,B]...]
        self.img = Image.open(imgpath) #whole slide image
        self.img_inked = Image.open(img_inkedpath) #marked whole slide image

    def _is_epi(self, src): #check if a black line passes the center (h/v) or diagonally
        margin = 4 #regard only the central 1/4-3/4 meaningful
        threshold = 40
        low = src.shape[0] // margin
        high = src.shape[0] - src.shape[0] // margin
        src_max = np.max(src, axis=2) #(299,299): max in [R,G,B]
        
        left = np.min(src_max[low:high, 0]) < threshold
        right = np.min(src_max[low:high, -1]) < threshold
        top = np.min(src_max[0, low:high]) < threshold
        bottom = np.min(src_max[-1, low:high]) < threshold

        horizontal = left and right
        vertical = top and bottom

        lt_side = min(np.min(src_max[:low, 0]), np.min(src_max[0, :low])) < threshold
        rt_side = min(np.min(src_max[0, high:]), np.min(src_max[:low, -1])) < threshold
        lb_side = min(np.min(src_max[high:, 0]), np.min(src_max[-1, :low])) < threshold
        rb_side = min(np.min(src_max[high:, -1]), np.min(src_max[-1, high:])) < threshold

        diagonal = (lt_side and rb_side) or (rt_side and lb_side)

        return horizontal or vertical or diagonal #boolean
    
    def _is_stroma(self, src): #check if no black line is in src
        src_max = np.max(src, axis=2)
        threshold = 40
        
        trgt = np.all(src_max > threshold) and np.mean(src) < 230 
        #and np.min(src_max) < 150
        return trgt
    #something is in the field
    
    def _is_marked(self, src): #check if src contains a mark
        color = {'red':(245, 30, 30), 'green':(30, 245, 30), 'blue':(30, 30, 245)}
        trgt = [] #[R,G,B](boolean)
        threshold = 2 #minimum necessary dots regarded as marked
        for i, clr in enumerate(color):
            c1 = np.where(src[:, :, (0+i)%3] >= color[clr][(0+i)%3], 1, 0)
            c2 = np.where(src[:, :, (1+i)%3] <= color[clr][(1+i)%3], 1, 0)
            c3 = np.where(src[:, :, (2+i)%3] <= color[clr][(2+i)%3], 1, 0)
            trgt.append((np.count_nonzero(c1 * c2 * c3) >= threshold)) 
        return trgt #list of booleans [R,G,B]
    
    def _pricrop_fortest(self):  #disused: reduce wsi size for test run
        left = 7000
        top = 5500
        right = 8000
        bottom = 6500
        self.img = self.img.crop((left, top, right, bottom))
        self.img_inked = self.img_inked.crop((left, top, right, bottom))
    
    def _save_stromalpatch(self, src): #Image
        global strpatch_filename
        src.save(STROMAFOLDER + strpatch_filename + '.jpg')
        strpatch_filename = str(int(strpatch_filename) + 1).zfill(6)
        
    
    def collect_patch(self, wsi_name, remove_similar=True, stroma=False,\
                      message=True, rotate=True, rate_epithelium=1,\
                      rate_stroma=50, balance_sample=True):
        #wsi_name: case number, remove_similar: discard overlapped patches
        #stroma: collect stromal patches to STROMAFOLDER (ToDo)
        #message: display num of saved patches remove_similar(True/False)
        #rotate: allign patches so that epithelium comes upside as much as possible
        #rate_epithelium: collect 1/rate_epithelium patches of normal epithelium
        #rate_stroma: collect 1/rate_stroma patches of stroma
        
        img_np = np.array(self.img)
        img_inked_np = np.array(self.img_inked)

        xn = (self.img_inked.size[0] - PIXEL)//STEP + 1
        yn = (self.img_inked.size[1] - PIXEL)//STEP + 1

        cnt_normepi = 1
        cnt_markepi = 1
        cnt_strpatch = 1
        
        for j in range(yn):
            for i in range(xn):
                
                inked_patch_np = img_inked_np[j * STEP:j * STEP + PIXEL,\
                                         i * STEP:i * STEP + PIXEL]

                if self._is_epi(inked_patch_np):
                    img_patch = Image.fromarray(img_np[j * STEP:j * STEP + PIXEL,\
                                                       i * STEP:i * STEP + PIXEL])
                    
                    mark = self._is_marked(inked_patch_np)
                    if sum(mark):       
                        self.img_patches.append(img_patch.resize((SIZE, SIZE)))
                        self.img_location.append((j * STEP, i * STEP))
                        self.patch_list.append((i, j))
                        self.mark_list.append(mark)
                        cnt_markepi += 1 #count marked patch number
                        
                    else:
                        if cnt_normepi % rate_epithelium == 0:
                            self.img_patches.append(img_patch.resize((SIZE, SIZE)))
                            self.img_location.append((j * STEP, i * STEP))
                            self.patch_list.append((i, j))
                            self.mark_list.append(mark)                                                 
                        cnt_normepi += 1 #count normal epithelial patch number
                    
                elif stroma:
                    if cnt_strpatch % rate_stroma == 0:                           
                        if self._is_stroma(inked_patch_np):
                            img_patch = Image.fromarray(img_np[j * STEP:j * STEP + PIXEL,\
                            i * STEP:i * STEP + PIXEL]).resize((SIZE, SIZE))
                            self._save_stromalpatch(img_patch)                      
                    cnt_strpatch += 1

        msg = len(self.patch_list) #stack for message

        if remove_similar: #discard overlapped patches
            self._remove_similar()
            
        if balance_sample:
            self._balance_class()

        if rotate:
            self._rotate_upsideup()
   
        self.save_label(wsi_name)
        self.save_patches()
        
#        print([i for i, x in enumerate(self.mark_list) if x == True]) #display marked patch indices
        if message:
            print('{}/{}'.format(len(self.patch_list), msg), end=' ')
            
    def _remove_similar(self):
        min_distance = 4  #actually sqrt(4) (=2 grids)
        if self.patch_list != []:
            #calculate distance between matrix elements
            all_diffs = np.expand_dims(self.patch_list, axis=1) -\
            np.expand_dims(self.patch_list, axis=0)
            distance = np.sum(all_diffs ** 2, axis=-1) #omitted sqrt
            distance_tri = np.triu(distance, k=1) #upper triangle marix
            
            #remove patches close (min_distance) to already taken patches
            selected_idx = [0] #always include the index of the first patch
            for i in range(1, len(self.patch_list)):
                if np.all(distance_tri[selected_idx, i] > min_distance):
                    selected_idx.append(i)

            #renew to selected patches
            self.patch_list = [self.patch_list[i] for i in selected_idx]
            self.img_location = [self.img_location[i] for i in selected_idx]
            self.img_patches = [self.img_patches[i] for i in selected_idx]
            self.mark_list = [self.mark_list[i] for i in selected_idx]
    
    def _balance_class(self):

        mark_n = len([1 for l in self.mark_list if l[0]==True or l[1]==True or l[2]==True])
        norm_n = len(self.mark_list) - mark_n
        
        total_idx = set([i for i in range(len(self.mark_list))])
        mark_idx = set([i for i, l in enumerate(self.mark_list)\
                    if l[0]==True or l[1]==True or l[2]==True])
        norm_idx = total_idx - mark_idx
        
        new_idx = sorted(list(total_idx))
        if norm_n > mark_n:
            selected_normidx = set(random.sample(norm_idx, mark_n))
            new_idx = sorted(list(mark_idx | selected_normidx))
        
        #renew to balanced patchset
        self.patch_list = [self.patch_list[i] for i in new_idx]
        self.img_patches = [self.img_patches[i] for i in new_idx]
        self.img_location = [self.img_location[i] for i in new_idx]
        self.mark_list = [self.mark_list[i] for i in new_idx]
    
        print('marked {} normal {} total {}'.format(mark_n, norm_n, len(new_idx)))
            
    def _rotate_upsideup(self): #put brightest side top
        angles = (0, 90, 180, 270)
        for i, patch in enumerate(self.img_patches):
            patch_np = np.array(patch)
            top = np.mean(patch_np[0, :, :])
            bottom = np.mean(patch_np[-1, :, :])
            left = np.mean(patch_np[:, 0, :])
            right = np.mean(patch_np[:, -1, :])
            brightest_side = np.argmax(np.array((top, right, bottom, left)))
            self.img_patches[i] = patch.rotate(angles[brightest_side])

    def display_patchpanel(self):#disused: display 200 (maximum*10) patches as a panel
        maximum = 20
        ylen = min(maximum, len(self.img_patches)//10 + 1)
        panel = Image.new('RGB', (2000, 200 * ylen))

        for i, img in enumerate(self.img_patches):
            panel.paste(img, (i%10 * 200, i//10 * 200))
            if i >= ylen * 10:
                break
        panel.show()

    def save_patches(self):
        global epipatch_filename
        patchnum = int(epipatch_filename)
        for i, patch in enumerate(self.img_patches):
            
            patch.save(SAVEFOLDER + str(patchnum + i).zfill(6) + '.jpg')

        epipatch_filename = str(patchnum + len(self.img_patches)).zfill(6)

    def save_label(self, wsi_name, epithelium=True):
        global df
        epi = int(epithelium)
        patchname = epipatch_filename
        for il, ml in zip(self.img_location, self.mark_list):
            
            s = pd.Series([wsi_name, il[0], il[1], epi, 0, 0, 0, 0, 0],\
                          index=df.columns, name=patchname)
            df = df.append(s)
            
            for i, clr in enumerate(COLORDICT):
                df.loc[patchname, COLORDICT[clr]] = int(ml[i])

            patchname = str(int(patchname) + 1).zfill(6)


def _check_consistency(src1, src2):
    #check consistency of img and marked img in the folder
    trgt = True
    if len(src1) != len(src2):
        trgt = False

    for s1, s2 in zip(src1, src2):
        if os.path.basename(s1).rstrip('_original.jpg') != \
        os.path.basename(s2).rstrip('_inked.jpg'):
            trgt = False
    return trgt #boolean

def determine_nextpatchname(folder):
    nextpatch = '0'.zfill(6)
    files = glob.glob(folder + '*.jpg')
    if files != []:
        putative_lastpatch = folder + str(len(files) - 1).zfill(6) + '.jpg'
        if putative_lastpatch in files: #check just in case
            nextpatch = str(len(files)).zfill(6)
        else:
            print('Number mismatch detected.')
    return nextpatch #str

def load_labelfile():    
    
    labelpath = glob.glob(SAVEFOLDER + '*.pkl')
    
    if len(labelpath) == 1:
        _df = pd.read_pickle(labelpath[0])
        print('label loaded')            
    elif len(labelpath) == 0:
        _df = pd.DataFrame(index=[], columns=COLUMNS)
        print('A new label has been generated')    
    else: #error
        print('Multiple labels in the assigned folder')
        _df = None        
    return _df #dataframe or None  

def save_labelfile():
    labelfilename = 'LabelNew' + str(datetime.datetime.now())[:10] + '.pkl'
    df.to_pickle(DESKTOP + labelfilename)
    print('The label has been saved as {} on desktop.'.format(labelfilename))

def _patchlabel(number=0):
    trgt = df.loc[str(number).zfill(6)]
    return trgt

##############
Image.MAX_IMAGE_PIXELS = 5000000000
DESKTOP = 'C:\\Users\\Moon\\Desktop\\'
PIXEL = 400 #image pixel taken from WSI
STEP = 100 #scan step (pixels)
SIZE = 299 #for InceptionV3 299x299 pix patches

COLORDICT = {'red':'HiGrade', 'green':'LoGrade', 'blue':'Invasion'} #dict of mark colors
BASEFOLDER = DESKTOP + 'wsi\\' #wsi both original and marked, and a label
SAVEFOLDER = DESKTOP + 'patches\\' #generated patches
STROMAFOLDER = DESKTOP + 'stroma_patches\\'
COLUMNS = ['Case', 'CordX', 'CordY', 'Epithelium', 'LoGrade', 'HiGrade', 'Invasion',\
            'Inflammation', 'Keratinization']

path1 = glob.glob(BASEFOLDER + '*_o.jpg')
path2 = glob.glob(BASEFOLDER + '*_i.jpg')

df = load_labelfile()


if df is None: #if more than 2 labels in BASEFOLDER
    sys.exit()

caselist = list(set(df['Case'].tolist()))

if _check_consistency(path1, path2): #path1 and path2 files must be completely matched.
    epipatch_filename = determine_nextpatchname(SAVEFOLDER)
    strpatch_filename = determine_nextpatchname(STROMAFOLDER)
    
    for p1, p2 in zip(path1, path2):
        wsi_name = os.path.basename(p1).rstrip('_o.jpg') #str
        print(wsi_name, end=': ')
        if wsi_name in caselist:
            print('included in the current patch collection.')
        else:
            im = ImagePatch(p1, p2)
            im.collect_patch(wsi_name, remove_similar=True, stroma=False,\
                             message=True, rate_epithelium=1, rate_stroma=25,\
                             balance_sample=False)
## remove_similar: remove overlapped patches
## stroma, rate_stroma: save stromal patches every (1/rate_stroma) in STROMAFOLDER 
## message: display number of saved patches/number of extracted patches
## rate_epithelium: save every (rate_epithelium) normal epithelium patches
## balance_sample: discard normal epithelium patches to make the normal/abnormal ratio of 1:1

save_labelfile()



        
          