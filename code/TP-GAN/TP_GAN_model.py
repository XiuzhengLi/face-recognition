#!/usr/bin/env python3
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np

from net_input_everything_featparts import *
from time import localtime, strftime
import random
import pickle
import subprocess
relu = tf.nn.relu
import scipy.misc
import shutil
from utils import pp, visualize, to_json

class TP_GAN(object):
    # 数据标准化处理
    def processor(self, images, reuse=False):
        #accept 3 channel images, output orginal 3 channels and 3 x 4 gradient map-> 15 channels
        with tf.variable_scope("processor") as scope:
            if reuse:
                scope.reuse_variables()
            input_dim = images.get_shape()[-1]
            gradientKernel = gradientweight()
            output_dim = gradientKernel.shape[-1]
            print("processor:", output_dim)
            k_hw = gradientKernel.shape[0]
            init = tf.constant_initializer(value=gradientKernel, dtype=tf.float32)
            w = tf.get_variable('w', [k_hw, k_hw, input_dim, output_dim],
                                initializer=init)
            conv = tf.nn.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME')
            #conv = conv * 2
            return tf.concat_v2([images, conv], 3)
        # 提取局部特征：左眼、右眼、鼻子、嘴巴
        def partRotator(self, images, name, batch_size=10, reuse=False):
        #HW 40x40, 32x40, 32x48
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            c0 = lrelu(conv2d(images, self.gf_dim, d_h=1, d_w=1, name="p_conv0"))
            c0r = resblock(c0, name="p_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, self.gf_dim*2, name="p_conv1"),name="p_bnc1"))
            #down1
            c1r = resblock(c1, name="p_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, self.gf_dim*4, name='p_conv2'),name="p_bnc2"))
            #down2
            c2r = resblock(c2, name="p_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, self.gf_dim*8, name='p_conv3'),name="p_bnc3"))
            #down3 5x5, 4x5, 4x6
            c3r = resblock(c3, name="p_conv3_res")
            c3r2 = resblock(c3r, name="p_conv3_res2")

            shape = c3r2.get_shape().as_list()
            d1 = lrelu(batch_norm(deconv2d(c3r2, [shape[0], shape[1] * 2, shape[2] * 2, self.gf_dim*4], name="p_deconv1"), name="p_bnd1"))
            #up1
            after_select_d1 = lrelu(batch_norm(conv2d(tf.concat_v2([d1, c2r], axis=3), self.gf_dim*4, d_h=1, d_w=1, name="p_deconv1_s"),name="p_bnd1_s"))
            d1_r = resblock(after_select_d1, name="p_deconv1_res")
            d2 = lrelu(batch_norm(deconv2d(d1_r, [shape[0], shape[1] * 4, shape[2] * 4, self.gf_dim*2], name="p_deconv2"), name="p_bnd2"))
            #up2
            after_select_d2 = lrelu(batch_norm(conv2d(tf.concat_v2([d2, c1r], axis=3), self.gf_dim*2, d_h=1, d_w=1, name="p_deconv2_s"),name="p_bnd2_s"))
            d2_r = resblock(after_select_d2, name="p_deconv2_res")
            d3 = lrelu(batch_norm(deconv2d(d2_r, [shape[0], shape[1] * 8, shape[2] * 8, self.gf_dim], name="p_deconv3"), name="p_bnd3"))
            #up3
            after_select_d3 = lrelu(batch_norm(conv2d(tf.concat_v2([d3, c0r], axis=3), self.gf_dim, d_h=1, d_w=1, name="p_deconv3_s"),name="p_bnd3_s"))
            d3_r = resblock(after_select_d3, name="p_deconv3_res")

            check_part = tf.nn.tanh(conv2d(d3_r, 3, d_h=1, d_w=1, name="p_check"))

        return d3_r, check_part

    # 局部特征组合
    def partCombiner(self, eyel, eyer, nose, mouth):
        '''
        x         y
        43.5823   41.0000
        86.4177   41.0000
        64.1165   64.7510
        47.5863   88.8635
        82.5904   89.1124
        this is the mean locaiton of 5 landmarks
        '''
        eyel_p = tf.pad(eyel, [[0,0], [int(41 - EYE_H / 2 - 1), int(IMAGE_SIZE - (41+EYE_H/2 - 1))], [int(44 - EYE_W / 2 - 1), int(IMAGE_SIZE - (44+EYE_W/2-1))], [0,0]])
        eyer_p = tf.pad(eyer, [[0,0], [int(41 - EYE_H / 2 - 1), int(IMAGE_SIZE - (41+EYE_H/2 - 1))], [int(86 - EYE_W / 2 - 1), int(IMAGE_SIZE - (86+EYE_W/2-1))], [0,0]])
        nose_p = tf.pad(nose, [[0,0], [int(65 - NOSE_H / 2 - 1), int(IMAGE_SIZE - (65+NOSE_H/2 - 1))], [int(64 - NOSE_W / 2 - 1), int(IMAGE_SIZE - (64+NOSE_W/2-1))], [0,0]])
        month_p = tf.pad(mouth, [[0,0], [int(89 - MOUTH_H / 2 - 1), int(IMAGE_SIZE - (89+MOUTH_H/2 - 1))], [int(65 - MOUTH_W / 2 - 1), int(IMAGE_SIZE - (65+MOUTH_W/2-1))], [0,0]])
        eyes = tf.maximum(eyel_p, eyer_p)
        eye_nose = tf.maximum(eyes, nose_p)
        return tf.maximum(eye_nose, month_p)

	def generator(self, images, batch_size, name = "generator", reuse = False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
        # imgs: input: IMAGE_SIZE x IMAGE_SIZE x CHANNEL
        # return labels: IMAGE_SIZE x IMAGE_SIZE x 3
        # U-Net structure, slightly different from the original on the location of relu/lrelu
            #128x128
            c0 = lrelu(conv2d(images, self.gf_dim, k_h=7, k_w=7, d_h=1, d_w=1, name="g_conv0"))
            c0r = resblock(c0, k_h=7, k_w=7, name="g_conv0_res")
            c1 = lrelu(batch_norm(conv2d(c0r, self.gf_dim, k_h=5, k_w=5, name="g_conv1"),name="g_bnc1"))
            #64x64
            c1r = resblock(c1, k_h=5, k_w=5, name="g_conv1_res")
            c2 = lrelu(batch_norm(conv2d(c1r, self.gf_dim*2, name='g_conv2'),name="g_bnc2"))
            #32x32
            c2r = resblock(c2, name="g_conv2_res")
            c3 = lrelu(batch_norm(conv2d(c2r, self.gf_dim*4, name='g_conv3'),name="g_bnc3"))
            #16x16
            c3r = resblock(c3, name="g_conv3_res")
            c4 = lrelu(batch_norm(conv2d(c3r, self.gf_dim*8, name='g_conv4'),name="g_bnc4"))
            #8x8
            c4r = resblock(c4, name="g_conv4_res")
            # c5 = lrelu(batch_norm(conv2d(c4r, self.gf_dim*8, name='g_conv5'),name="g_bnc5"))
            # #4x4
            # #2x2
            # c6r = resblock(c6,k_h=2, k_w=2, name="g_conv6_res")
            c4r2 = resblock(c4r, name="g_conv4_res2")
            c4r3 = resblock(c4r2, name="g_conv4_res3")
            c4r4 = resblock(c4r3, name="g_conv4_res4")
            c4r4_l = tf.reshape(c4r4,[batch_size, -1])
            c7_l = linear(c4r4_l, output_size=512,scope='feature', bias_start=0.1, with_w=True)[0]
            c7_l_m = tf.maximum(c7_l[:, 0:256], c7_l[:, 256:])

            return c0r, c1r, c2r, c3r, c4r4, c7_l_m

    def decoder(self, feat128, feat64, feat32, feat16, feat8, featvec,
                g128_images_with_code, g64_images_with_code, g32_images_with_code,
                eyel, eyer, nose, mouth, c_eyel, c_eyer, c_nose, c_mouth, batch_size = 10, name="decoder", reuse = False):
        sel_feat_capacity = self.gf_dim
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            initial_all = tf.concat_v2([featvec, self.z], 1)
            initial_8 = relu(tf.reshape(linear(initial_all, output_size=8*8*self.gf_dim,scope='initial8', bias_start=0.1, with_w=True)[0],
                                        [batch_size, 8, 8, self.gf_dim]))
            initial_32 = relu(deconv2d(initial_8, [batch_size, 32, 32, self.gf_dim // 2], d_h=4, d_w=4, name="initial32"))
            initial_64 = relu(deconv2d(initial_32, [batch_size, 64, 64, self.gf_dim // 4], name="initial64"))
            initial_128 = relu(deconv2d(initial_64, [batch_size, 128, 128, self.gf_dim // 8], name="initial128"))

            before_select8 = resblock(tf.concat_v2([initial_8, feat8], 3), k_h=2, k_w=2, name = "select8_res_1")
            #selection T module
            reconstruct8 = resblock(resblock(before_select8, k_h=2, k_w=2, name="dec8_res1"), k_h=2, k_w=2, name="dec8_res2")

            #selection F module
            reconstruct16_deconv = relu(batch_norm(deconv2d(reconstruct8, [batch_size, 16, 16, self.gf_dim*8], name="g_deconv16"), name="g_bnd1"))
            before_select16 = resblock(feat16, name = "select16_res_1")
            reconstruct16 = resblock(resblock(tf.concat_v2([reconstruct16_deconv, before_select16], 3), name="dec16_res1"), name="dec16_res2")

            reconstruct32_deconv = relu(batch_norm(deconv2d(reconstruct16, [batch_size, 32, 32, self.gf_dim*4], name="g_deconv32"), name="g_bnd2"))
            before_select32 = resblock(tf.concat_v2([feat32, g32_images_with_code, initial_32], 3), name = "select32_res_1")
            reconstruct32 = resblock(resblock(tf.concat_v2([reconstruct32_deconv, before_select32], 3), name="dec32_res1"), name="dec32_res2")
            img32 = tf.nn.tanh(conv2d(reconstruct32, 3, d_h=1, d_w=1, name="check_img32"))

            reconstruct64_deconv = relu(batch_norm(deconv2d(reconstruct32, [batch_size, 64, 64, self.gf_dim*2], name="g_deconv64"), name="g_bnd3"))
            before_select64 = resblock(tf.concat_v2([feat64, g64_images_with_code, initial_64], 3), k_h=5, k_w=5, name = "select64_res_1")
            reconstruct64 = resblock(resblock(tf.concat_v2([reconstruct64_deconv, before_select64,
                                                            tf.image.resize_bilinear(img32, [64,64])], 3), name="dec64_res1"), name="dec64_res2")
            img64 = tf.nn.tanh(conv2d(reconstruct64, 3, d_h=1, d_w=1, name="check_img64"))

            reconstruct128_deconv = relu(batch_norm(deconv2d(reconstruct64, [batch_size, 128, 128, self.gf_dim], name="g_deconv128"), name="g_bnd4"))
            before_select128 = resblock(tf.concat_v2([feat128, initial_128, g128_images_with_code],3), k_h = 7, k_w = 7, name = "select128_res_1")
            reconstruct128 = resblock(tf.concat_v2([reconstruct128_deconv, before_select128,
                                                    self.partCombiner(eyel, eyer, nose, mouth),
                                                    self.partCombiner(c_eyel, c_eyer, c_nose, c_mouth),
                                                    tf.image.resize_bilinear(img64, [128,128])], 3), k_h=5, k_w=5, name="dec128_res1")
            reconstruct128_1 = lrelu(batch_norm(conv2d(reconstruct128, self.gf_dim, k_h=5, k_w=5, d_h=1, d_w=1, name="recon128_conv"), name="recon128_bnc"))
            reconstruct128_1_r = resblock(reconstruct128_1, name="dec128_res2")
            reconstruct128_2 = lrelu(batch_norm(conv2d(reconstruct128_1_r, self.gf_dim/2, d_h=1, d_w=1, name="recon128_conv2"),name="recon128_bnc2"))
            img128 = tf.nn.tanh(conv2d(reconstruct128_2, 3, d_h=1, d_w=1, name="check_img128"))

            return img128, img64, img32, img32, img32, img128, img64, img32

    
