# -*- coding:utf8 -*-
from extract_cnn import BuildModel
import numpy as np
import h5py






import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''
    def __init__(self):
        self.path = '../dataset'  #表示需要命名处理的文件夹




    def rename(self):
        filelist = os.listdir(self.path) #获取文件路径

        etra_feats = [[]] * 6
        etra_name = [[]] * 6
        # etra_name = [[]*5]
        for n,item in enumerate(filelist):

            #find the dirct
            fulldirct = os.path.join(os.path.abspath(self.path), item)
            if os.path.isdir(fulldirct):  # 入参需要是绝对路径
                #get every files
                img_list = []
                for f in os.listdir(fulldirct):
                    if f.endswith('.jpg'):
                        img_list.append(os.path.join(fulldirct,f))

                feats = []
                names = []
                model = BuildModel()

                for i, img_path in enumerate(img_list):
                    norm_feat = model.extract_feat(img_path)
                    img_name = os.path.split(img_path)[1]
                    feats.append(norm_feat)
                    names.append(img_name)
                    # print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(fulldirct)))
                feats = np.array(feats)
                etra_feats[n] = feats
                etra_name[n] = names
                print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(fulldirct)))



                # directory for storing extracted features
                # output = args["index"]
        output = "model/features.h5"

        print("--------------------------------------------------")
        print("      writing feature extraction results ...")
        print("--------------------------------------------------")
        h5f = h5py.File(output, 'w')
        #for m in range(5):
        h5f.create_dataset('qi-baishi_f', data=etra_feats[1])
        h5f.create_dataset('ding-yanyong_f', data=etra_feats[2])
        h5f.create_dataset('shitao_f', data=etra_feats[3])
        h5f.create_dataset('huang-yongyu_f', data=etra_feats[4])
        h5f.create_dataset('xu-beihong_f', data=etra_feats[5])

        #h5f.create_dataset('dataset_6', data=etra_name[1:6])
        h5f.create_dataset('qi-baishi', data=etra_name[1])
        h5f.create_dataset('ding-yanyong', data=etra_name[2])
        h5f.create_dataset('shitao', data=etra_name[3])
        h5f.create_dataset('huang-yongyu', data=etra_name[4])
        h5f.create_dataset('xu-beihong', data=etra_name[5])

        '''
            h5f.create_dataset('dataset_2', data=etra_feats[1])
            h5f.create_dataset('dataset_3', data=etra_feats[2])
            h5f.create_dataset('dataset_4', data=etra_feats[3])
            h5f.create_dataset('dataset_5', data=etra_feats[4])

            h5f.create_dataset('dataset_6', data=etra_name[0])
            h5f.create_dataset('dataset_6', data=etra_name[1])
            h5f.create_dataset('dataset_6', data=etra_name[2])
            h5f.create_dataset('dataset_6', data=etra_name[3])
            h5f.create_dataset('dataset_6', data=etra_name[4])
        '''



        h5f.close()

        '''
                if item.endswith('.jpg'):  #初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
                    src = os.path.join(os.path.abspath(self.path), item)
                    #dst = os.path.join(os.path.abspath(self.path), ''+str(i) + '.jpg')#处理后的格式也为jpg格式的，当然这里可以改成png格式
                    dst = os.path.join(os.path.abspath(self.path), name +'0' + format(str(i), '0>2s') + '.jpg')    #这种情况下的命名格式为0000000.jpg形式，可以自主定义想要的格式
                    try:
                        os.rename(src, dst)
                        print ('converting %s to %s ...' % (src, dst))
                        i = i + 1
                    except:
                        continue

        '''

        #print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()