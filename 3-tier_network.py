# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils_3tier import *
from model import *
from glob import glob
from skimage import color,filters
import cv2

sess = tf.Session()

input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_high_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_r')
input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')
input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

[R_decom, I_decom] = DecomNet_simple(input_decom)
decom_output_R = R_decom
decom_output_I = I_decom
output_r = Restoration_net(input_low_r, input_low_i)
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'Illumination_adjust_net' in var.name]
var_restoration = [var for var in tf.trainable_variables() if 'Restoration_net' in var.name]

saver_Decom = tf.train.Saver(var_list = var_Decom)
saver_adjust = tf.train.Saver(var_list=var_adjust)
saver_restoration = tf.train.Saver(var_list=var_restoration)

decom_checkpoint_dir ='./checkpoint/decom_net_train/'
ckpt_pre=tf.train.get_checkpoint_state(decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No decomnet checkpoint!')

checkpoint_dir_adjust = './checkpoint/illumination_adjust_net_train/'
ckpt_adjust=tf.train.get_checkpoint_state(checkpoint_dir_adjust)
if ckpt_adjust:
    print('loaded '+ckpt_adjust.model_checkpoint_path)
    saver_adjust.restore(sess,ckpt_adjust.model_checkpoint_path)
else:
    print("No adjust pre model!")

checkpoint_dir_restoration = './checkpoint/Restoration_net_train/'
ckpt=tf.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No restoration pre model!")

### frame_pre() #동영상 to 이미지
    ### 파일 한개만 넣을거면 정렬 안해도 될듯
    
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob('./test/1.mp4')
eval_low_data_name.sort()

for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)

    video_to_image(eval_low_data_name[idx])


###load eval data
    
eval_low_data2 = []
eval_img_name2 =[]
eval_low_data_name2 = glob('./frame_pre/*.png')
eval_low_data_name2.sort()
eval_low_data_name2 = np.asarray(eval_low_data_name2) #list to numpy
#print(eval_low_data_name2.shape) #(139,)

print('image loading ... ')
for idx in range(len(eval_low_data_name2)):
    [_, name] = os.path.split(eval_low_data_name2[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name2.append(name)
    eval_low_im2 = load_images(eval_low_data_name2[idx])
    eval_low_data2.append(eval_low_im2)
print('image loading  success... ')


sample_dir = './frame_post/' #evaluate_pc
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("Start evalating!")
print(time.strftime('%Y-%m-%d %p %I:%M:%S', time.localtime(time.time())))
for idx in range(len(eval_low_data2)):
    print(idx)
    name = eval_img_name2[idx]
    input_low = eval_low_data2[idx]
    input_low_eval = np.expand_dims(input_low, axis=0)
    h, w, _ = input_low.shape

    decom_r_low, decom_i_low = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low_eval})
    #save_images(os.path.join(sample_dir, '%s_decom_r_pc.png' % (name)), decom_r_low)
    #save_images(os.path.join(sample_dir, '%s_decom_i_pc.png' % (name)), decom_i_low)

    restoration_r = sess.run(output_r, feed_dict={input_low_r: decom_r_low, input_low_i: decom_i_low})
    #save_images(os.path.join(sample_dir, '%s_restoration_pc.png' % (name)), restoration_r)

    ratio = 4.0
    i_low_data_ratio = np.ones([h, w]) * (ratio)
    i_low_ratio_expand = np.expand_dims(i_low_data_ratio, axis=2)
    i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)
    adjust_i = sess.run(output_i, feed_dict={input_low_i: decom_i_low, input_low_i_ratio: i_low_ratio_expand2})
    #save_images(os.path.join(sample_dir, '%s_adjust_pc.png' % (name)), adjust_i)

    decom_r_sq = np.squeeze(decom_r_low)
    r_gray = color.rgb2gray(decom_r_sq)
    r_gray_gaussion = filters.gaussian(r_gray, 3)
    low_i = np.minimum((r_gray_gaussion * 2) ** 0.5, 1)
    low_i_expand_0 = np.expand_dims(low_i, axis=0)
    low_i_expand_3 = np.expand_dims(low_i_expand_0, axis=3)
    result_denoise = restoration_r * low_i_expand_3
    fusion4 = result_denoise * adjust_i

    fusion2 = decom_i_low * input_low_eval + (1 - decom_i_low) * fusion4
    save_images(os.path.join(sample_dir, '%s.png' % (name)), fusion2)
    
print(time.strftime('%Y-%m-%d %p %I:%M:%S', time.localtime(time.time())))


###'./frame_post/'에 저장된 이미지 to 동영상

inputpath = 'C:/Users/96dks/Desktop/KinD/frame_post/*.png'
outpath =  './result/filename.mp4' # .avi 확장자도 가능
fps = 30

eval_low_data = []
eval_img_name =[]
ifiles = glob(inputpath)
ifiles.sort()
ifilesl = len(ifiles)

for idx in range(ifilesl):
    [_, name] = os.path.split(ifiles[idx])
    suffix = name[name.find('.') + 1:] #png
    name = name[:name.find('.')] #0
    eval_img_name.append(name) #['0']
    
    image_to_video(ifiles, outpath, fps, ifilesl)
    #ifiles[idx] == C:/Users/96dks/Desktop/KinD/frame_post\0.png

print(time.strftime('%Y-%m-%d %p %I:%M:%S', time.localtime(time.time())))
