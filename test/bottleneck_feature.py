#!/usr/bin/python
# coding:utf8

# fine-tune网络的后面几层.Fine-tune以一个预训练好的网络为基础,在新的数据集上重新训练一小部分权重
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import numpy as np

train_data = np.load(open('/Users/momo/Documents/mhf/IP/test/model/bottleneck_features_train.npy'))
train_labels = np.array([0]*3000 + [1]*3000)
print (train_labels)
validation_data = np.load(open('/Users/momo/Documents/mhf/IP/test/model/bottleneck_features_validation.npy'))
validation_labels = np.array([0]*1200 + [1]*1200)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_data=(validation_data,validation_labels))
model.save_weights('bottleneck_fc_model.h5')
