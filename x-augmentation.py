# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:39:38 2017

@author: neusoft
"""
# ############################
# #使用keras 数据增强处理的验证 试验

import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image as imageGenerator
import glob

CUR_PATH = 'x-image'

SAVE_PATH = 'x-gen'

GENERATORCOUNTER = 5

GEN_PATH = 'x-gen'

# ############################
# # fill_mode 填充模式(constant,nearest默认,reflect,wrap)
# # 当设置为constant时 还有一个可选参数cval 代表使用某个固定数值的颜色来进行填充
# # datagen = imageGenerator.ImageDataGenerator(fill_mode='wrap', zoom_range=[4, 4])

# ############################
# # featurewise 处理后 视觉上图像会稍微变暗
def gen_featurewise(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
    
    print('generator ...')
    
    # #save_prefix 保存的文件的前缀 在save_to_dir被指定后有效
    # #前提 图像文件必须放到CUR_PATH下的子目录下 
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    print(gen_data)
    # #生成x张图
    for i in range(icount):
        gen_data.next()


# ############################
# # samplewise 处理后 视觉上图像会变成 类x光机图像形式
def gen_samplewise(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# #zca_whtening zca白化处理 视觉上图像会变成 灰白图像
def gen_whtening(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(zca_whitening=True)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # rotation range 随机旋转图像 在 [0, 指定角度] 范围内进行随机角度旋转
def gen_rotation(icount=GENERATORCOUNTER,rotation_range=30):
    datagen = imageGenerator.ImageDataGenerator(rotation_range=30)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # width_shift_range & height_shift_range 水平平移,垂直平移,平移距离在[0,最大平移距离]区间内
# # 尽量不要设置太大的数值
def gen_shiftrange(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(width_shift_range=0.5,height_shift_range=0.5)

    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # shear_range 错切变换 就是某个坐标不变 另一个坐标按比例平移
def gen_shearrange(icount=GENERATORCOUNTER,shear_range=0.5):
    datagen = imageGenerator.ImageDataGenerator(shear_range)

    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # zoom_range 让图片在长或宽的方向进行放大 [zoom_range] 或[width_zoom_range, height_zoom_range]
def gen_zoom(icount=GENERATORCOUNTER,zoom_range=0.5):
    datagen = imageGenerator.ImageDataGenerator(zoom_range)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # channel_shift_range 改变图片的整体的颜色 但并不能单独改变图片某一元素的颜色，如黑色小狗不能变成白色小狗
def gen_channelshift(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(channel_shift_range=10)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # horizontal_flip 随机对图片执行水平翻转操作 不一定对所有图片都会执行水平翻转
def gen_horizontalflip(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(horizontal_flip=True)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# # vertical_flip 随机对图片执行上下翻转操作 不一定对所有图片都会执行上下翻转
def gen_verticalflip(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(vertical_flip=True)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()

# ############################
# # rescale 对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行  用肉眼看是没有任何区别的
def gen_rescale(icount=GENERATORCOUNTER):
    datagen = imageGenerator.ImageDataGenerator(rescale= 1/255, width_shift_range=0.1)
    
    gen_data = datagen.flow_from_directory(CUR_PATH,
                                       batch_size=1,
                                       shuffle=False,
                                       save_to_dir=SAVE_PATH,
                                       save_prefix='gen',
                                       save_format='jpeg',
                                       target_size=(224, 224))
    # #生成x张图
    for i in range(icount):
        gen_data.next()


# ############################
# # 
print('start ...')
# #gen_featurewise()
print('end ...')

# #gen_samplewise()

# #gen_whtening()

# #gen_rotation(GENERATORCOUNTER,190)

# #可用
# #gen_shiftrange()
# #可用
# #gen_shearrange(GENERATORCOUNTER,0.2)

gen_zoom(GENERATORCOUNTER,0.2)

# #gen_channelshift()

# #gen_horizontalflip()

# #gen_verticalflip()

# #gen_rescale()

# #找到本地生成图，把所有图打印到同一张figure上
name_list = glob.glob(GEN_PATH+'/*')
fig = plt.figure()

for i in range(len(name_list)):
    img = Image.open(name_list[i])
    sub_img = fig.add_subplot(331 + i)
    sub_img.imshow(img)

# #plt.pause(30)

plt.show()


