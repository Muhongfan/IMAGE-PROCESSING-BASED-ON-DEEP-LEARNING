# -*- coding:utf8 -*-
import os

path = '/Users/momo/Documents/mhf/IP_PROJECT/data'  # 获取当前路径

num_dirs = 0  # 路径下文件夹数量
num_files = 0  # 路径下文件数量(包括文件夹)
num_files_rec = 0  # 路径下文件数量,包括子文件夹里的文件数量，不包括空文件夹
less = 0
more = 0
for root, dirs, files in os.walk(path):  # 遍历统计
    for each in files:
        if each[-2:] == '.o':
            print(root, dirs, each)
        num_files_rec += 1
    for name in dirs:
        num_dirs += 1
        #num_in_file=0
        subname = os.path.join(root,name)
        num_in_file = 0
        for subroot, subdirs, subfiles in os.walk(subname):
            for num_subfile in subfiles:
                num_in_file += 1
            if 0<num_in_file and num_in_file<50:
                less+=1
            else:
                more += 1

            print(str(num_in_file) +"      "+  "%3s" %name)
print(less)
print(more)

'''
for fn in os.listdir(path):
    num_files += 1
    print fn

'''
