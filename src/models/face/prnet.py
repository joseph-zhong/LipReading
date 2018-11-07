#!/usr/bin/env python3
"""
prnet.py
---



"""

import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope

from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp

import src.utils.utility as _util

class PRN:
  ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
  Args:
      is_dlib(bool, optional): If true, dlib is used for detecting faces.
      prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
  '''

  def __init__(self, is_dlib=False):

    # resolution of input and output image size.
    self.resolution_inp = 256
    self.resolution_op = 256

    # ---- load detectors
    if is_dlib:
      import dlib
      detector_path = _util.getRelWeightsPath('dlib', 'mmod_human_face_detector.dat')
      self.face_detector = dlib.cnn_face_detection_model_v1(detector_path)

    # ---- load PRN
    self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
    prn_path = _util.getRelWeightsPath('prnet', 'net/256_256_resfcn256_weight')
    assert os.path.isfile(prn_path + '.data-00000-of-00001'), "please download PRN trained model first."
    self.pos_predictor.restore(prn_path)

    # uv file: 2 x 68
    self.uv_kpt_ind = np.loadtxt(_util.getRelWeightsPath('prnet', 'uv', 'uv_kpt_ind.txt')).astype(np.int32)
    #  get kpt: get valid vertices in the pos map
    self.face_ind = np.loadtxt(_util.getRelWeightsPath('prnet', 'uv', 'face_ind.txt')).astype(np.int32)
    # ntri x 3.
    self.triangles = np.loadtxt(_util.getRelWeightsPath('prnet', 'uv', 'triangles.txt')).astype(np.int32)

    self.uv_coords = self.generate_uv_coords()

  def generate_uv_coords(self):
    resolution = self.resolution_op
    uv_coords = np.meshgrid(range(resolution), range(resolution))
    uv_coords = np.transpose(np.array(uv_coords), [1, 2, 0])
    uv_coords = np.reshape(uv_coords, [resolution ** 2, -1])
    uv_coords = uv_coords[self.face_ind, :]
    uv_coords = np.hstack((uv_coords[:, :2], np.zeros([uv_coords.shape[0], 1])))
    return uv_coords

  def dlib_detect(self, image):
    return self.face_detector(image, 1)

  def net_forward(self, image):
    ''' The core of out method: regress the position map of a given image.
    Args:
        image: (256,256,3) array. value range: 0~1
    Returns:
        pos: the 3D position map. (256, 256, 3) array.
    '''
    return self.pos_predictor.predict(image)

  def process(self, input, image_info=None):
    ''' process image with crop operation.
    Args:
        input: (h,w,3) array or str(image path). image value range:1~255.
        image_info(optional): the bounding box information of faces. if None, will use dlib to detect face.

    Returns:
        pos: the 3D position map. (256, 256, 3).
    '''
    if isinstance(input, str):
      try:
        image = imread(input)
      except IOError:
        print("error opening file: ", input)
        return None
    else:
      image = input

    if image.ndim < 3:
      image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

    if image_info is not None:
      if np.max(image_info.shape) > 4:  # key points to get bounding box
        kpt = image_info
        if kpt.shape[0] > 3:
          kpt = kpt.T
        left = np.min(kpt[0, :])
        right = np.max(kpt[0, :])
        top = np.min(kpt[1, :])
        bottom = np.max(kpt[1, :])
      else:  # bounding box
        bbox = image_info
        left = bbox[0]
        right = bbox[1]
        top = bbox[2]
        bottom = bbox[3]
      old_size = (right - left + bottom - top) / 2
      center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
      size = int(old_size * 1.6)
    else:
      detected_faces = self.dlib_detect(image)
      if len(detected_faces) == 0:
        print('warning: no detected face')
        return None

      d = detected_faces[
        0].rect  ## only use the first detected face (assume that each input image only contains one face)
      left = d.left()
      right = d.right()
      top = d.top()
      bottom = d.bottom()
      old_size = (right - left + bottom - top) / 2
      center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
      size = int(old_size * 1.58)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
      [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image = image / 255.
    cropped_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

    # run our net
    # st = time()
    cropped_pos = self.net_forward(cropped_image)
    # print 'net time:', time() - st

    # restore
    cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, [self.resolution_op, self.resolution_op, 3])

    return pos

  def get_landmarks(self, pos):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[self.uv_kpt_ind[1, :], self.uv_kpt_ind[0, :], :]
    return kpt

  def get_vertices(self, pos):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [self.resolution_op ** 2, -1])
    vertices = all_vertices[self.face_ind, :]

    return vertices

  def get_colors_from_texture(self, texture):
    '''
    Args:
        texture: the texture map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    all_colors = np.reshape(texture, [self.resolution_op ** 2, -1])
    colors = all_colors[self.face_ind, :]

    return colors

  def get_colors(self, image, vertices):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    '''
    [h, w, _] = image.shape
    vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
    vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:, 1], ind[:, 0], :]  # n x 3

    return colors

def resBlock(x, num_outputs, kernel_size=4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm,
    scope=None):
  assert num_outputs % 2 == 0  # num_outputs must be divided by channel_factor(2 here)
  with tf.variable_scope(scope, 'resBlock'):
    shortcut = x
    if stride != 1 or x.get_shape()[3] != num_outputs:
      shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
        activation_fn=None, normalizer_fn=None, scope='shortcut')
    x = tcl.conv2d(x, num_outputs / 2, kernel_size=1, stride=1, padding='SAME')
    x = tcl.conv2d(x, num_outputs / 2, kernel_size=kernel_size, stride=stride, padding='SAME')
    x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

    x += shortcut
    x = normalizer_fn(x)
    x = activation_fn(x)
  return x


class resfcn256(object):
  def __init__(self, resolution_inp=256, resolution_op=256, channel=3, name='resfcn256'):
    self.name = name
    self.channel = channel
    self.resolution_inp = resolution_inp
    self.resolution_op = resolution_op

  def __call__(self, x, is_training=True):
    with tf.variable_scope(self.name) as scope:
      with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
        with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
            normalizer_fn=tcl.batch_norm,
            biases_initializer=None,
            padding='SAME',
            weights_regularizer=tcl.l2_regularizer(0.0002)):
          size = 16
          # x: s x s x 3
          se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1)  # 256 x 256 x 16
          se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2)  # 128 x 128 x 32
          se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1)  # 128 x 128 x 32
          se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2)  # 64 x 64 x 64
          se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1)  # 64 x 64 x 64
          se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2)  # 32 x 32 x 128
          se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1)  # 32 x 32 x 128
          se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2)  # 16 x 16 x 256
          se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1)  # 16 x 16 x 256
          se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2)  # 8 x 8 x 512
          se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1)  # 8 x 8 x 512

          pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1)  # 8 x 8 x 512
          pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2)  # 16 x 16 x 256
          pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
          pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1)  # 16 x 16 x 256
          pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2)  # 32 x 32 x 128
          pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
          pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1)  # 32 x 32 x 128
          pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2)  # 64 x 64 x 64
          pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64
          pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1)  # 64 x 64 x 64

          pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2)  # 128 x 128 x 32
          pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1)  # 128 x 128 x 32
          pd = tcl.conv2d_transpose(pd, size, 4, stride=2)  # 256 x 256 x 16
          pd = tcl.conv2d_transpose(pd, size, 4, stride=1)  # 256 x 256 x 16

          pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
          pd = tcl.conv2d_transpose(pd, 3, 4, stride=1)  # 256 x 256 x 3
          pos = tcl.conv2d_transpose(pd, 3, 4, stride=1,
            activation_fn=tf.nn.sigmoid)  # , padding='SAME', weights_initializer=tf.random_normal_initializer(
          # 0, 0.02))

          return pos

  @property
  def vars(self):
    return [var for var in tf.global_variables() if self.name in var.name]


class PosPrediction():
  def __init__(self, resolution_inp=256, resolution_op=256):
    # -- hyper settings
    self.resolution_inp = resolution_inp
    self.resolution_op = resolution_op
    self.MaxPos = resolution_inp * 1.1

    # network type
    self.network = resfcn256(self.resolution_inp, self.resolution_op)

    # net forward
    self.x = tf.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
    self.x_op = self.network(self.x, is_training=False)
    self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

  def restore(self, model_path):
    tf.train.Saver(self.network.vars).restore(self.sess, model_path)

  def predict(self, image):
    pos = self.sess.run(self.x_op,
      feed_dict={self.x: image[np.newaxis, :, :, :]})
    pos = np.squeeze(pos)
    return pos * self.MaxPos

  def predict_batch(self, images):
    pos = self.sess.run(self.x_op,
      feed_dict={self.x: images})
    return pos * self.MaxPos









