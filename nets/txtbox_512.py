
""" 
This framework is based on SSD_tensorlow(https://github.com/balancap/SSD-Tensorflow)
Add descriptions
"""

import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import textbox_common

slim = tf.contrib.slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
TextboxParams = namedtuple('TextboxParameters', 
										['img_shape',
										 'num_classes',
										 'feat_layers',
										 'feat_shapes',
										 'normalizations',
										 'gamma',
										 ])

class TextboxNet(object):

	default_params = TextboxParams(
		img_shape=(512, 512),
		num_classes=2,
		feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11'],
		feat_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2)],
		normalizations=[20, -1, -1, -1, -1, -1],
		gamma=1.5
		)

	def __init__(self, params=None):
		"""
		Init the Textbox net with some parameters. Use the default ones
		if none provided.
		"""
		if isinstance(params, TextboxParams):
			self.params = params
		else:
			self.params = self.default_params

	# ======================================================================= #
	def net(self, inputs,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_512',
			use_batch=False):
		"""
		Text network definition.
		"""
		r = text_net(inputs,
					feat_layers=self.params.feat_layers,
					normalizations=self.params.normalizations,
					is_training=is_training,
					dropout_keep_prob=dropout_keep_prob,
					reuse=reuse,
					use_batch=use_batch,
					scope=scope)

		return r

	def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Network arg_scope.
		"""
		return ssd_arg_scope(weight_decay, data_format=data_format)


	def anchors(self, dtype=np.float32):
		"""Compute the default anchor boxes, given an image shape.
		"""
		return textbox_common.textbox_anchor_all_layers(self.params.img_shape,
									  self.params.feat_shapes,
									  self.params.gamma,
									  0.5,
									  dtype)

	def bboxes_encode(self, cord, anchors, num,
					  scope='text_bboxes_encode'):
		"""Econde the groudtruth Lables, offsets and links
		"""
		return textbox_common.tf_text_bboxes_encode(cord, anchors, num,)

	def bboxes_decode(self, feat_localizations, anchors, scope='ssd_bboxes_decode'):
		"""
		Encode labels and bounding boxes.
		"""
		# TODO

	def detected_bboxes(self, predictions, localisations,
						select_threshold=None, nms_threshold=0.5,
						clipping_bbox=None, top_k=400, keep_top_k=200):
		"""
		Get the detected bounding boxes from the SSD network output.
		"""
		# Select top_k bboxes from predictions, and clip
		# TODO


	def losses(self, logits, localisations,linkslogits,
			   glocalisations, gscores, glinks,
			   negative_ratio=3.,
			   use_hard_neg=True,
			   alpha1=1.,
			   alpha2=1.,
			   label_smoothing=0.,
			   scope='text_box_loss'):
		"""Define the SSD network losses.
		"""
		return text_losses(logits, localisations, linkslogits,
						  glocalisations, gscores, glinks,
						  negative_ratio=negative_ratio,
						  alpha1=alpha1,
						  alpha2=alpha2,
						  label_smoothing=label_smoothing,
						  scope=scope)



def text_net(inputs,
			feat_layers=TextboxNet.default_params.feat_layers,
			normalizations=TextboxNet.default_params.normalizations,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			use_batch=True,
			scope='text_box_512'):
	batch_norm_params = {
	  # Decay for the moving averages.
	  'decay': 0.997,
	  'epsilon': 0.001,
	  'is_training': is_training
	}
	end_points = {}
	with tf.variable_scope(scope, 'text_box_300', [inputs], reuse=reuse):
		# Original VGG-16 blocks.
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		end_points['conv1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		# Block 2.
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		end_points['conv2'] = net # 150,150 128
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		# Block 3. # 75 75 256
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		end_points['conv3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3',padding='SAME')
		# Block 4. # 38 38 512
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		end_points['conv4'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		# Block 5. # 19 19 512
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		end_points['conv5'] = net
		net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5',padding='SAME')

		net = conv2d(net, 1024, [3,3], scope='conv6',rate = 6, use_batch=use_batch, batch_norm_params= batch_norm_params)
		end_points['conv6'] = net

		net = conv2d(net, 1024, [1, 1], scope='conv7',use_batch=use_batch, batch_norm_params= batch_norm_params)
		end_points['conv7'] = net

		end_point = 'conv8'
		with tf.variable_scope(end_point):
			net = conv2d(net, 256, [1, 1], scope='conv1x1',use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 512, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)	
		end_points[end_point] = net
		end_point = 'conv9'
		with tf.variable_scope(end_point):
			net = conv2d(net, 128, [1, 1], scope='conv1x1', use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 256, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)
		end_points[end_point] = net
		end_point = 'conv10'
		with tf.variable_scope(end_point):
			net = conv2d(net, 128, [1, 1], scope='conv1x1',use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 256, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)
		end_points[end_point] = net
		end_point = 'conv11'
		with tf.variable_scope(end_point):
			#net = conv2d(net, 128, [1, 1], scope='conv1x1',use_batch=use_batch, batch_norm_params=batch_norm_params)
			net = conv2d(net, 256, [3, 3], stride=2, scope='conv3x3',use_batch=use_batch, batch_norm_params=batch_norm_params)
		end_points[end_point] = net


		print end_points
		# Prediction and localisations layers.
		linkslogits = []
		logits = []
		localisations = []
		for i, layer in enumerate(feat_layers):
			with tf.variable_scope(layer + '_box'):
				p, loc, link = text_multibox_layer(layer,
										  end_points[layer],
										  normalizations[i],
										  is_training=is_training,
										  use_batch=use_batch)
			logits.append(p)
			localisations.append(loc)
			linkslogits.append(link)

		return localisations, logits, linkslogits, end_points

def conv2d(inputs, out, kernel_size, scope,stride=1,activation_fn=tf.nn.relu, 
			padding = 'SAME',rate = 1,use_batch=False, batch_norm_params={}):
	if use_batch:
		net = slim.conv2d(inputs, out, kernel_size, stride=stride ,scope=scope, normalizer_fn=slim.batch_norm, 
			  normalizer_params=batch_norm_params, activation_fn=None ,padding = padding, rate = rate)
	else:
		net = slim.conv2d(inputs, out, kernel_size, stride=stride, scope=scope, activation_fn=activation_fn,padding = padding, rate=rate)
	return net


def text_multibox_layer(layer,
					   inputs,
					   normalization=-1,
					   is_training=True,
					   use_batch=False):
	"""
	
	"""
	batch_norm_params = {
	  # Decay for the moving averages.
	  'decay': 0.9997,
	  # epsilon to prevent 0s in variance.
	  'epsilon': 0.001,
	  'is_training': is_training
	}
	net = inputs
	if normalization > 0:
		net = custom_layers.l2_normalization(net, scaling=True)
	# Number of anchors.
	num_box = 1
	num_classes = 2
	# Location.
	num_loc_pred = 5


	loc_pred = conv2d(net, num_loc_pred, [3, 3], activation_fn=None, padding = 'SAME',
						   scope='conv_loc',use_batch=False, batch_norm_params=batch_norm_params)

	loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [5])

	# Class prediction.
	scores_pred = num_classes
	sco_pred = conv2d(net, scores_pred, [3, 3], activation_fn=None, padding = 'SAME',
						   scope='conv_cls',use_batch=use_batch, batch_norm_params=batch_norm_params)

	sco_pred = tf.reshape(sco_pred, tensor_shape(sco_pred, 4)[:-1] + [num_classes])

	# links prediction.
	scores_pred = 12 *  num_classes
	links_pred = conv2d(net, scores_pred, [3, 3], activation_fn=None, padding = 'SAME',
						   scope='conv_link',use_batch=use_batch, batch_norm_params=batch_norm_params)

	links_pred = tf.reshape(links_pred, tensor_shape(links_pred, 4)[:-1] + [12,num_classes])
	return sco_pred, loc_pred, links_pred


def tensor_shape(x, rank=3):
	"""Returns the dimensions of a tensor.
	Args:
	  image: A N-D Tensor of shape.
	Returns:
	  A list of dimensions. Dimensions that are statically known are python
		integers,otherwise they are integer scalar tensors.
	"""
	if x.get_shape().is_fully_defined():
		return x.get_shape().as_list()
	else:
		static_shape = x.get_shape().with_rank(rank).as_list()
		dynamic_shape = tf.unstack(tf.shape(x), rank)
		return [s if s is not None else d
				for s, d in zip(static_shape, dynamic_shape)]




def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
	"""Defines the VGG arg scope.

	Args:
	  weight_decay: The l2 regularization coefficient.

	Returns:
	  An arg_scope.
	"""
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_regularizer=slim.l2_regularizer(weight_decay),
						#weights_initializer=tf.truncated_normal_initializer(stddev=0.03, seed = 1000),
						weights_initializer=tf.contrib.layers.xavier_initializer(),
						biases_initializer=tf.zeros_initializer()):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
							padding='SAME',
							data_format=data_format):
			with slim.arg_scope([custom_layers.pad2d,
								 custom_layers.l2_normalization,
								 custom_layers.channel_to_last],
								data_format=data_format) as sc:
				return sc



# =========================================================================== #
# Text loss function.
# =========================================================================== #
def text_losses(logits, localisations, linkslogits,
			   glocalisations, gscores, glinks,
			   negative_ratio=3.,
			   alpha1=1.,
			   alpha2=1.,
			   label_smoothing=0.,
			   scope=None):
	with tf.name_scope(scope, 'text_loss'):
		alllogits = []
		alllocalization = []
		alllinkslogits = []
		allglocalization = []
		allgscores = []
		allglinks = []
		for i in range(len(logits)):
			alllogits.append(tf.reshape(logits[i], [-1, 2]))
			allgscores.append(tf.reshape(gscores[i], [-1]))
			allglinks.append(tf.reshape(glinks[i], [-1,12]))
			alllinkslogits.append(tf.reshape(linkslogits[i], [-1,12,2]))
			allglocalization.append(tf.reshape(glocalisations[i], [-1,5]))
			alllocalization.append(tf.reshape(localisations[i], [-1,5]))

		alllogits = tf.concat(alllogits, 0)
		allgscores = tf.concat(allgscores, 0)
		allglinks = tf.concat(allglinks, 0)
		alllinkslogits = tf.concat(alllinkslogits, 0)
		alllocalization =tf.concat(alllocalization, 0)
		allglocalization =tf.concat(allglocalization, 0)

		pmask = tf.cast(allgscores ,tf.bool)
		ipmask = tf.cast(pmask ,tf.int32)
		n_pos = tf.reduce_sum(ipmask)+1
		num = tf.ones_like(allgscores)
		n = tf.reduce_sum(num)
		fpmask = tf.cast(pmask , tf.float32)
		nmask = tf.cast(1- allgscores, tf.bool)

		## segment score loss
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alllogits,labels=ipmask)
		cross_pos = tf.losses.compute_weighted_loss(loss, fpmask)
		loss_neg = tf.where(pmask,
						   tf.cast(tf.zeros_like(ipmask),tf.float32),
						   loss)
		loss_neg_flat = tf.reshape(loss_neg, [-1])
		n_neg = tf.minimum(3*n_pos, tf.cast(n,tf.int32))
		val, idxes = tf.nn.top_k(loss_neg_flat, k=n_neg)
		minval = val[-1]
		nmask = tf.logical_and(nmask, loss_neg >= minval)

		fnmask = tf.cast(nmask, tf.float32)
		cross_neg = tf.losses.compute_weighted_loss(loss, fnmask)

		## localization loss
		weights = tf.expand_dims(fpmask, axis=-1)
		l_loc = custom_layers.abs_smooth(alllocalization - allglocalization)
		l_loc = tf.losses.compute_weighted_loss(l_loc, weights)


		## links score loss
		pmask_l = tf.cast(allglinks, tf.bool)
		ipmask_l = tf.cast(pmask_l, tf.int32)
		n_pos_l = tf.reduce_sum(ipmask_l)+1
		num_l = tf.ones_like(ipmask_l)
		n_l = tf.reduce_sum(num_l)
		fpmask_l = tf.cast(pmask_l , tf.float32)
		nmask_l = tf.cast(1- allglinks, tf.bool)

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=alllinkslogits,labels=ipmask_l)
		l_cross_pos = tf.losses.compute_weighted_loss(loss, fpmask_l)
		loss_neg = tf.where(pmask_l,
						   tf.cast(tf.zeros_like(ipmask_l),tf.float32),
						   loss)
		loss_neg_flat = tf.reshape(loss_neg, [-1])
		n_neg = tf.minimum(3*n_pos_l, tf.cast(n_l,tf.int32))
		val, idxes = tf.nn.top_k(loss_neg_flat, k=n_neg)
		minval = val[-1]
		nmask_l = tf.logical_and(nmask_l, loss_neg >= minval)

		fnmask_l = tf.cast(nmask_l, tf.float32)
		l_cross_neg = tf.losses.compute_weighted_loss(loss, fnmask_l)



		with tf.name_scope('total'):
				# Add to EXTRA LOSSES TF.collection
				total_cross = tf.add(cross_pos, cross_neg, 'cross_entropy')
				total_cross_l = tf.add(l_cross_pos, l_cross_neg,'cross_entropy_links')
				#total_cross = tf.identity(total_cross, name = 'total_cross')
				n_pos = tf.identity(n_pos, name = 'num_of_positive')
				n_pos_l = tf.identity(n_pos_l, name = 'num_of_positive_links')
				cross_pos = tf.identity(cross_pos, name = 'cross_pos')
				cross_neg = tf.identity(cross_neg, name = 'cross_neg')
				l_cross_neg = tf.identity(l_cross_neg, name = 'l_cross_neg')
				l_cross_pos = tf.identity(l_cross_pos, name = 'l_cross_pos')
				l_loc = tf.identity(l_loc, name = 'l_loc')

				tf.add_to_collection('EXTRA_LOSSES', n_pos)
				tf.add_to_collection('EXTRA_LOSSES', n_pos_l)
				tf.add_to_collection('EXTRA_LOSSES', l_cross_pos)
				tf.add_to_collection('EXTRA_LOSSES', l_cross_neg)
				tf.add_to_collection('EXTRA_LOSSES', cross_pos)
				tf.add_to_collection('EXTRA_LOSSES', cross_neg)
				tf.add_to_collection('EXTRA_LOSSES', l_loc)
				tf.add_to_collection('EXTRA_LOSSES', total_cross)
				tf.add_to_collection('EXTRA_LOSSES', total_cross_l)

				total_loss = tf.add_n([alpha1*l_loc, total_cross, alpha2*total_cross_l], 'total_loss')
				tf.add_to_collection('EXTRA_LOSSES', total_loss)

	return total_loss

