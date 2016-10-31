#coding:utf-8

import os
from PIL import Image 
import numpy as np

#��ȡ�ļ���mnist�µ�42000��ͼƬ��ͼƬΪ�Ҷ�ͼ������Ϊ1ͨ����
#����ǽ���ɫͼ��Ϊ����,��1�滻Ϊ3,ͼ���С28*28
def load_data():
    data = np.empty((42000,1,28,28),dtype="float32")
    label = np.empty((42000,),dtype="uint8")

    img_list = os.listdir("./mnist")
    count = len(img_list)
    for i in range(count):
        img = Image.open("./mnist/"+img_list[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = arr
        label[i] = int(img_list[i].split('.')[0])
    return data,label

if __name__ == '__main__':
    data,label = load_data()
    print data.shape
    print label.shape
    print label[0]
    print label[10000]

