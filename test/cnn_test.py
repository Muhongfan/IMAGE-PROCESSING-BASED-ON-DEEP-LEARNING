#!/usr/bin/python
# coding:utf8

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/momo/Documents/mhf/IP/data/train'
validation_data_dir = '/Users/momo/Documents/mhf/IP/data/validation'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 100
batch_size = 40

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# 建立模型
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))
# 编译
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)
# 从图片中直接产生数据和标签
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(150, 150),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(150, 150),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps  =nb_validation_samples // batch_size)

test_dir = '/Users/momo/Documents/mhf/IP/dataset'
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print("Test accurency:", test_acc)
print("Test loss:", test_loss)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epchs = range(1, len(acc) + 1)


plt.plot(epchs, acc, 'bo', label='Training accurency')
plt.plot(epchs, val_acc, 'b', label='Validation accurency')
plt.title('Training and Validation accurency')
plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)

plt.figure()

plt.plot(epchs, loss, 'bo', label='Training loss')
plt.plot(epchs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)

plt.show()


# 保存整个模型
model.save('model_new.hdf5')

# 保存模型的权重
model.save_weights('model_weights_new.h5')

# 保存模型的结构
json_string = model.to_json()
open('model_to_json_new.json', 'w').write(json_string)
yaml_string = model.to_yaml()
open('model_to_yaml_new.yaml', 'w').write(json_string)