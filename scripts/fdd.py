import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import cv2
import numpy as np
import time
import datetime
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import requests

URL = 'http://101.132.74.147:8081/data/setDataList'


def fdd():
    model = MSCNN((224, 224, 3))
    model.load_weights('../models/model_weights.h5')

    # 当前十个设备
    devices_name = ['Surveillance_camera1', 'Surveillance_camera2', 'Surveillance_camera3', 'Surveillance_camera4',
                    'Surveillance_camera5', 'Surveillance_camera6', 'Surveillance_camera7', 'Surveillance_camera8',
                    'Surveillance_camera9', 'Surveillance_camera10']
    gc = []

    # 设备初始化
    for i in range(len(devices_name)):
        gc.append(Get_Capture(devices_name[i]))

    nowMinute = datetime.datetime.now().minute
    while True:
        # 对比当前时间 若时间更新则进行数据更新
        if datetime.datetime.now().minute != nowMinute:
            # 延时5s 防止网络问题导致图片上传不成功
            time.sleep(5)
            nowMinute = datetime.datetime.now().minute
            print('-------------------------------')
            print(nowMinute)
            # 待发送数据
            sendData = []
            for i in range(len(devices_name)):
                # 加载图片
                img = gc[i].loadImg(datetime.datetime.now().minute)
                if img is not None:
                    # 图像预处理
                    img = cv2.resize(img, (224, 224)) / 255.
                    img = np.expand_dims(img, axis=0)
                    # 预测人数
                    dmap = np.squeeze(model.predict(img), axis=-1)  # 降维
                    dmap = cv2.GaussianBlur(dmap, (15, 15), 0)  # 高斯模糊
                    gc[i].num = int(np.sum(dmap))
                else:
                    gc[i].num = -1
                # 打包数据
                sendData.append(jsonData(gc[i].id, gc[i].latitude, gc[i].longitude, gc[i].num).__dict__)
            print(sendData)
            res = requests.post(url=URL, json=sendData)
            print(res)
        time.sleep(1)


class Get_Capture:
    def __init__(self, dirname):
        self.folder = '../data/'
        self.dirname = dirname
        self.Image = None
        self.num = -1
        # 读取位置信息
        f = open(self.folder + self.dirname + "/Location.txt", "r")
        Str_ = f.read().split()
        self.latitude = float(Str_[0])
        self.longitude = float(Str_[1])
        self.id = Str_[2]
        f.close()

    def loadImg(self, index):
        if os.path.exists(self.folder + self.dirname + '/Capture ({}).jpg'.format(index)):
            self.Image = cv2.imread(self.folder + self.dirname + '/Capture ({}).jpg'.format(index))
            return self.Image
        else:
            return None


class jsonData:
    def __init__(self, id, latitude, longitude, num):
        self.cameraId = id
        self.latitude = latitude
        self.longitude = longitude
        self.num = num


def MSB(filter_num):
    def f(x):
        params = {
            'strides': 1,
            'activation': 'relu',
            'padding': 'same',
            'kernel_regularizer': l2(5e-4)
        }
        x1 = Conv2D(filters=filter_num, kernel_size=(9, 9), **params)(x)
        x2 = Conv2D(filters=filter_num, kernel_size=(7, 7), **params)(x)
        x3 = Conv2D(filters=filter_num, kernel_size=(5, 5), **params)(x)
        x4 = Conv2D(filters=filter_num, kernel_size=(3, 3), **params)(x)
        x = concatenate([x1, x2, x3, x4])
        x = BatchNormalization()(x)
        return x

    return f


def MSB_mini(filter_num):
    def f(x):
        params = {
            'strides': 1,
            'activation': 'relu',
            'padding': 'same',
            'kernel_regularizer': l2(5e-4)
        }
        x2 = Conv2D(filters=filter_num, kernel_size=(7, 7), **params)(x)
        x3 = Conv2D(filters=filter_num, kernel_size=(5, 5), **params)(x)
        x4 = Conv2D(filters=filter_num, kernel_size=(3, 3), **params)(x)
        x = concatenate([x2, x3, x4])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    return f


def MSCNN(input_shape=(224, 224, 3)):
    # 构建模型
    input_tensor = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(9, 9), strides=1, padding='same', activation='relu')(input_tensor)
    # block2
    x = MSB(4 * 16)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # block3
    x = MSB(4 * 32)(x)
    x = MSB(4 * 32)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = MSB_mini(3 * 64)(x)
    x = MSB_mini(3 * 64)(x)

    x = Conv2D(1000, (1, 1), activation='relu', kernel_regularizer=l2(5e-4))(x)

    x = Conv2D(1, (1, 1))(x)
    x = Activation('sigmoid')(x)
    x = Activation('relu')(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model


if __name__ == '__main__':
    fdd()
