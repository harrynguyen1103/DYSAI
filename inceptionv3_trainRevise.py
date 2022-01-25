# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:17:06 2019

@author: DeepBlue
"""

import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.objectives import categorical_crossentropy
import keras.backend as K
import tensorflow as tf
from sklearn.utils import class_weight

#def analyze_testimage():
#    testimagedir = 'C:\\Users\\Rascal\\Desktop\\test_image\\'
#    model_location = desktop + 'incept07091717.h5'
#    testmodel = load_model(model_location)
#    file_names = glob.glob(testimagedir + '*.jpg')
#
#    for filename in file_names:
#
#        img = Image.open(filename).resize((img_size, img_size))
#        var = np.array(img)
#        var = var/255.
#        var = np.expand_dims(var, axis=0)
#
#        features = testmodel.predict(var)
#        print(features)
#        
#        plt.imshow(Image.open(filename))
#        plt.show()

def calculate_classweight():
    labels = []
    labels.extend([0] * len(glob.glob(train_dir + 'nrm\\*.jpg')))
    labels.extend([1] * len(glob.glob(train_dir + 'lg\\*.jpg')))
    labels.extend([2] * len(glob.glob(train_dir + 'hg\\*.jpg')))
    labels.extend([3] * len(glob.glob(train_dir + 'scc\\*.jpg')))
    dst = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    return dst

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def custom_loss():
    def loss_function(y_true, y_pred, axis=-1):

        
#        default setting: no modification
        param = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],\
                        [0, 0, 0, 1]])
        #allow low/high error, high/SCC error to some extent
        # param = np.array([[1, 0, 0, 0], [0, 0.9, 0.1, 0], [0, 0.1, 0.9, 0],\
        #                  [0, 0, 0, 1]])
#        param = np.array([[1, 0, 0, 0], [0, 0.8, 0.2, 0], [0, 0.2, 0.8, 0],\
#                         [0, 0, 0, 1]]) 
#        param = np.array([[1, 0, 0, 0], [0, 0.7, 0.3, 0], [0, 0.3, 0.7, 0],\
#                         [0, 0, 0, 1]])        
        
        x = tf.constant(param, dtype='float32')
        
        y_pred /= tf.reduce_sum(y_pred, axis, True)
        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        
        y_true = K.dot(y_true, x)
               
#        return - tf.reduce_sum(y_true * tf.log(y_pred), axis)
        return - tf.reduce_sum(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred), axis)
    return loss_function

##########Main
K.clear_session()
img_size = 299
desktop = 'C:\\Users\\Moon\\Desktop\\153_25_25_new\\'
train_dir = desktop + 'train\\'
test_dir = desktop +  'test\\'
# desktop1 = 'C:\\Users\\Moon\\Desktop\\'
 
#Savefile name
savepath = str(datetime.datetime.today())[5:16]
savepath = savepath.replace('-', '').replace(' ', '').replace(':', '')
savepath = desktop + 'incept' + savepath + '.h5'

##calculate class_weight
classweight = calculate_classweight()

#calculate epoch number
train_epoch = len(glob.glob(train_dir + '*\\*.jpg')) 
test_epoch = len(glob.glob(test_dir + '*\\*.jpg'))
train_epoch //= 320
test_epoch //=320

train_datagen = ImageDataGenerator(rescale=1./ 255,
                                   featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   rotation_range=45,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   zoom_range=0,
                                   fill_mode='constant',
                                   cval = 1.)

test_datagen = ImageDataGenerator(rescale=1./ 255)


# train_datagen.fit(x_train)

######## Multiclass train
train_generator = train_datagen.flow_from_directory(train_dir,\
        target_size=(img_size, img_size),
        classes=['nrm', 'lg', 'hg', 'scc'],
        batch_size=50, class_mode="categorical")
test_generator = test_datagen.flow_from_directory(test_dir,\
        target_size=(img_size, img_size),
        classes=['nrm', 'lg', 'hg', 'scc'],
        batch_size=50, class_mode="categorical")

#model = InceptionV3(weights='imagenet', include_top=True)
#print('model structure: ', model.summary())

#Inception v3モデルの読み込み。最終層は読み込まない
base_model = InceptionV3(weights='imagenet', include_top=False)
#最終層の設定
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, kernel_initializer="glorot_uniform",
                    activation="softmax", kernel_regularizer=l2(.0005))(x)
## multi classification usually uses softmax, but sigmoid was used intentionally
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:249]:
    layer.trainable = False
    if layer.name.startswith('batch_normalization'):
        layer.trainable = True
for layer in model.layers[249:]:
    layer.trainable = True

#opt = SGD(lr=.01, momentum=.9)
opt = Adam()
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss=custom_loss(), metrics=['accuracy'])
#model.summary()
checkpointer = ModelCheckpoint(filepath=savepath, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit_generator(train_generator,
                              epochs=30,
                              steps_per_epoch=train_epoch,
                              validation_data=test_generator,
                              validation_steps=test_epoch,
                              class_weight=classweight,
                              verbose=1,
                              callbacks=[reduce_lr, checkpointer])

## Draw the learning curve
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc)+1)
#
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accurracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
##Use this to make a save file.
plt.savefig(desktop + 'learning_curve.png')
#plt.show()

