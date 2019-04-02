# -*- coding:utf8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
import numpy as np

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/momo/Documents/mhf/IP/data/train'
validation_data_dir = '/Users/momo/Documents/mhf/IP/data/validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 50
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# （1）导入bottleneck_features数据
train_data = np.load(open('/Users/momo/Documents/mhf/IP/test/model/bottleneck_features_train.npy','rb'))
# the features were saved in order, so recreating the labels is easy
train_labels = np.array([0] * 500 + [1] * 500 + [2] * 500 + [3] * 500 + [4] * 500)  # matt,打标签

validation_data = np.load(open('/Users/momo/Documents/mhf/IP/test/model/bottleneck_features_validation.npy','rb'))
validation_labels = np.array([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100)  # matt,打标签

# （2）设置标签，并规范成Keras默认格式
train_labels = keras.utils.to_categorical(train_labels, 5)
validation_labels = keras.utils.to_categorical(validation_labels, 5)

# （3）写“小网络”的网络结构
model = Sequential()
#train_data.shape[1:]
model.add(Flatten(input_shape=(4,4,512)))# 4*4*512
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))  # 二分类
model.add(Dense(5, activation='softmax'))  # matt,多分类
#model.add(Dense(1))
#model.add(Dense(5))
#model.add(Activation('softmax'))

# （4）设置参数并训练
model.compile(loss='categorical_crossentropy',
# matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=epochs, batch_size=batch_size,
          validation_data=(validation_data, validation_labels))

model.save_weights('bottleneck_fc_model.h5')

