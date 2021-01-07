from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminator(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, s=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4


def generator_resnet(image, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(_x, dim, ks=3, s=1, _name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(_x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=_name + '_c1'), _name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=_name + '_c2'), _name + '_bn2')
            return y + _x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim * 4, _name='g_r1')
        r2 = residule_block(r1, options.gf_dim * 4, _name='g_r2')
        r3 = residule_block(r2, options.gf_dim * 4, _name='g_r3')
        r4 = residule_block(r3, options.gf_dim * 4, _name='g_r4')
        r5 = residule_block(r4, options.gf_dim * 4, _name='g_r5')
        r6 = residule_block(r5, options.gf_dim * 4, _name='g_r6')
        r7 = residule_block(r6, options.gf_dim * 4, _name='g_r7')
        r8 = residule_block(r7, options.gf_dim * 4, _name='g_r8')
        r9 = residule_block(r8, options.gf_dim * 4, _name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        return tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def abs_combine_ssim(in_, target, g_window, norm_const):
    return tf.reduce_mean(tf.abs(in_ - target) * g_window)/norm_const


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def gaussian_window(size, channels, sigma):
    gaussian = np.arange(-(size/2), size/2)
    gaussian = np.exp(-1. * gaussian**2/(2*sigma**2))
    gaussian = np.outer(gaussian, gaussian.reshape((size, 1)))
    gaussian = gaussian/np.sum(gaussian)
    gaussian = np.reshape(gaussian, (1, size, size, 1))
    gaussian = np.tile(gaussian, (1, 1, 1, channels))
    return gaussian
