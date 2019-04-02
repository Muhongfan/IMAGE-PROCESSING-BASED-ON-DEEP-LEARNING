#coding=UTF-8
import keras
from django.shortcuts import render
from .form import UploadImageForm
from .models import Image

from extract_cnn import VGG
from extract_def_cnn import BuildModel

import h5py
import os, sys

import numpy as np
import PIL.Image as image  # 加载pil的包
import matplotlib.pyplot as plt




def index(request):
    """imgs upload"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = Image(photo=request.FILES['image'])
            picture.save()

            dict = imageclassify(picture)
            return render(request, 'result.html', {'picture': picture, 'dict': dict})

    else:
        form = UploadImageForm()

    return render(request, 'index.html', {'form': form})


def store_pic(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = Image(photo=request.FILES['image'])
            picture.save()
    return render(request, 'index.html',{'form': form})

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''



def imageclassify(picture):
    keras.backend.clear_session()

    h5f = h5py.File('/Users/momo/Documents/mhf/IP/imageupload/model/feature_VGG.h5', 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    pic_name=picture.photo.name
    #pic = pic_name[0:6]

    queryDir = '/Users/momo/Documents/mhf/IP/imageupload/static/images/%s' % (pic_name) #+ ".jpg"

    model = VGG()
    #model = BuildModel()

    queryVec = model.extract_feat(queryDir)

    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    maxres =2
    imlist = [imgNames[index] for i,index in enumerate(rank_ID[1:maxres+1])]

    dict = []
    for i in range(0,2):
        dict.append([imlist[i],rank_score[i+1]])

    return dict












