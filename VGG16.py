#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 01:18:26 2020

@author: aycaburcu
"""

from keras import layers
from keras import models 
from keras import optimizers
import json

"""
test edicez
"""
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing import image
import numpy as np

import os

model=VGG16(weights='imagenet',
            include_top=True,
            input_shape=(224,224,3))

model.summary()

# bunu bir test görüntüsü üzerinde kullanalım 
list_name=os.listdir('/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/VGG16/test_img')
test_dir='test_img/'

for x,name in enumerate(list_name):
    path=test_dir+name
    Giris1=image.load_img(path,
                          target_size=(224,224))
    #Numpy dizisine dönüştür
    Giris=image.img_to_array(Giris1)
    #Görüntüuü ağa uygula
    y=model.predict(Giris.reshape(1,224,224,3))

    json_file = open('/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/VGG16/imagenet_class_index.json')
    etiketler=json.load(json_file)
    json_file.close()
    with open('/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/VGG16/imagenet_class_index.json') as dosya:
        etiketler=json.load(dosya)

    # #En yüksek tahmin sınıfını bul
    # tahmin_indeks=np.argmax(y)
    # tahmin_yuzde=y[0][tahmin_indeks]*100

    # #Belirlenen etiket
    # sinif=etiketler[str(tahmin_indeks)][1]

    # print(sinif)
    # print(tahmin_indeks)
    # print(tahmin_yuzde)

    #top5 accuracy
    #kucukten buyuge dogru tahminleri sıraladı
    tahminler=np.argsort(y[0])
    enust_5=tahminler[999:994:-1]#top5 accuracy
#tahminleri yazdir

    plt.subplot(1,4,x+1)
    plt.xticks([])
    plt.yticks([])

    say=1
    enust_list=[]
    for indeks in enust_5:
        id1,sinif=etiketler[str(indeks)]
        say=say+1
        enust_list.append(['sınıf',say,'=',sinif,"%",(100*y[0][indeks]).astype('int')])
    plt.xlabel("{}\n{}\n{}\n{}\n{}".format(* enust_list).replace(",","").replace("'",""))
    

    #örnek görüntüyü göster
    plt.imshow(Giris1)




