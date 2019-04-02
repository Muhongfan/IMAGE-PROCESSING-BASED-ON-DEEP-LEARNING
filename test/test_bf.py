# -*- coding:utf8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import keras
from keras import backend as K

import numpy as np

datagen = ImageDataGenerator(rescale=1./255)

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/momo/Documents/mhf/IP/data/train'
validation_data_dir = '/Users/momo/Documents/mhf/IP/data/validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 100
batch_size = 80


# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')
#get the symbolic outputs of each "key" layer (we gave them unique names).
#layer_dict = dict([(layer.name, layer) for layer in model.layers])

#（2）灌入pre-model的权重
model.load_weights('/Users/momo/Documents/mhf/IP/test/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# 训练集图像生成器
generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

#　验证集图像生成器
generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)


#（3）得到bottleneck feature
bottleneck_features_train = model.predict_generator(generator, 50)
# 核心，steps是生成器要返回数据的轮数，每个epoch含有500张图片，与model.fit(samples_per_epoch)相对
np.save(open('/Users/momo/Documents/mhf/IP/test/model/bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(generator, 10)
# 与model.fit(nb_val_samples)相对，一个epoch有800张图片，验证集
np.save(open('/Users/momo/Documents/mhf/IP/test/model/bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

