# -*- coding: utf-8 -*-

import h5py
import numpy as np


f = h5py.File('/Users/momo/Documents/mhf/testTF/test/model/features.h5', 'r')  # 打开h5文件
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    #print(f[key].value)