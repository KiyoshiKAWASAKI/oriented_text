
import tensorflow as tf
import numpy as np
import math
import tf_extended as tfe





# =========================================================================== #
# TensorFlow implementation of Text Boxes encoding / decoding.
# =========================================================================== #

def tf_text_bboxes_encode_layer(cord,
							 anchors_layer, num, gamma,
							 dtype=tf.float32):
		"""Get the groudtruth for label and offset
	
		"""
		yref, xref, href, wref = anchors_layer
	
		# Initialize tensors...
		shape = (yref.shape[0], yref.shape[1])
		feat_scores = tf.zeros(shape, dtype=dtype)
		feat_w    = tf.zeros(shape, dtype=dtype)
		feat_h    = tf.zeros(shape, dtype=dtype)
		feat_x    = tf.zeros(shape, dtype=dtype)
		feat_y    = tf.zeros(shape, dtype=dtype)
		feat_theta= tf.zeros(shape, dtype=dtype)

		def positive_anchors(bbox):

				vec_box = tf.stack([[bbox[1,1] - bbox[0,1], bbox[1,0] - bbox[0,0]],
														 [bbox[3,1] - bbox[0,1], bbox[3,0] - bbox[0,0]],
														 [bbox[3,1] - bbox[2,1], bbox[3,0] - bbox[2,0]],
														 [bbox[1,1] - bbox[2,1], bbox[1,0] - bbox[2,0]]])
						
				vec_anchor = tf.stack([[yref - bbox[0,1], xref - bbox[0,0]],
														 [yref - bbox[0,1], xref - bbox[0,0]],
														 [yref - bbox[2,1], xref - bbox[2,0]],
														 [yref - bbox[2,1], xref - bbox[2,0]]])
				vec_anchor = tf.transpose(vec_anchor, perm=(2,3,0,1))
				postive_anchor = tf.reduce_sum(vec_anchor * vec_box,axis=-1)
				postive_anchor = tf.cast(postive_anchor > 0, tf.int8)
				postive_anchor = tf.equal(tf.reduce_sum(postive_anchor, -1), tf.constant(4,dtype=tf.int8))

				height = tf.sqrt(tf.square(bbox[3,1] - bbox[0,1]) + tf.square(bbox[3,0] - bbox[0,0]))
				postive_anchor = tf.logical_and(postive_anchor, tf.reduce_max([href/height, height/href]) < 1.5)
				return postive_anchor

		def condition(i, feat_scores,
									feat_x, feat_y, feat_w, feat_h, feat_theta):
				"""Condition: check label index.
				"""
				#r = tf.less(i, tf.shape(bboxes)[0])
				r = tf.less(i, num)
				return r

		def body(i, feat_scores,feat_x, feat_y, feat_w, feat_h, feat_theta):
				"""Body: update feature labels, scores and bboxes.
				Follow the original SSD paper for that purpose:
					- assign values when jaccard > 0.5;
					- only update if beat the score of other bboxes.
				"""
				# Jaccard score.

				bbox = cord[i]
				angle = tf.atan((bbox[3,1] - bbox[2,1])/tf.abs(bbox[2,0] - bbox[3,0]))
				height = tf.sqrt(tf.square(bbox[3,1] - bbox[0,1])+tf.square(bbox[3,0] - bbox[0,0]))
				rotate_matrix = tf.stack([-tf.sin(angle), tf.cos(angle)])
				a_cord = tf.transpose(tf.stack([bbox[0,0] - xref, bbox[0,1]-yref]),perm=(1,2,0))
				d_cord = tf.transpose(tf.stack([bbox[3,0] - xref, bbox[3,1]-yref]),perm=(1,2,0))
				y_a = tf.reduce_sum(a_cord*rotate_matrix,axis=-1) + yref
				y_d = tf.reduce_sum(a_cord*rotate_matrix,axis=-1) + yref
				ys = (y_a + y_d)/2

				mask = positive_anchors(bbox)
				score = height/href
				mask = tf.logical_and(mask, tf.greater(score, feat_scores))
				imask = tf.cast(mask, tf.int64)
				fmask = tf.cast(mask, dtype)
				feat_scores = tf.where(mask, tf.ones_like(feat_scores,dtype=dtype)*score, feat_scores)
				feat_theta  = tf.where(mask, tf.ones_like(feat_scores,dtype=dtype)*angle, feat_theta)
				feat_h = tf.where(mask, tf.ones_like(feat_scores,dtype=dtype)*tf.log(height/gamma), feat_h)
				feat_y = tf.where(mask, ys, feat_y)


				return [i+1, feat_scores,
								feat_x, feat_y, feat_w, feat_h, feat_theta]



		i = 0
		[i,feat_scores,feat_x, feat_y, feat_w, feat_h, feat_theta] = \
																	tf.while_loop(condition, body,
																	[i, feat_scores,
																	feat_x, feat_y, feat_w, feat_h, feat_theta])

		feat_localizations = tf.stack([feat_x, feat_y, feat_w, feat_h, feat_theta], axis=-1)
		feat_label = tf.cast(feat_scores > 0, tf.int32)
		return feat_localizations, feat_label

def in_layer_links_encode(label, ):
		shape = label.get_shape().as_list()
		full_label = tf.pad(label, [[1,1],[1,1]], "CONSTANT")

		label_flat = tf.reshape(label,[-1,1])

		indices = tf.expand_dims(tf.range(shape[0]*shape[1],dtype=tf.int32), -1)


		com_label = tf.concat([indices,label_flat],axis=1)


		def update_in(x,full_label=full_label, shape=shape):
				i = tf.div(x[0],shape[0])
				j = tf.mod(x[0],shape[0])
				return tf.cond(tf.less(x[1], 1), 
											 lambda :tf.ones([8,],dtype=tf.int32), 
											 lambda: in_layer(full_label, i+1, j+1))


		def in_layer(full_label, i, j):
				indice = [[i-1,j-1],[i-1,j],[i-1,j+1],
									[i,j-1],[i,j+1],
									[i+1,j-1],[i+1,j],[i+1,j+1]]
				values = [full_label[k[0],k[1]] for k in indice]
				return tf.stack(values)

		return tf.map_fn(update_in,com_label)


def cross_layer_links_encode(label_cross, label):

		shape = label.get_shape().as_list()
		label_flat = tf.reshape(label,[-1,1])

		indices = tf.expand_dims(tf.range(shape[0]*shape[1],dtype=tf.int32), -1)


		com_label = tf.concat([indices,label_flat],axis=1)


		def update_cross(x,label=label_cross, shape=shape):
				i = tf.div(x[0],shape[0])
				j = tf.mod(x[0],shape[0])
				return tf.cond(tf.less(x[1], 1), 
											 lambda :tf.ones([4,],dtype=tf.int32), 
											 lambda :cross_layer(label, i, j))


		def cross_layer(label,i,j):
				indice = [[2*i,2*j],[2*i,2*j+1],[2*i+1,2*j],[2*i+1,2*j+1]]
				values = [label[k[0],k[1]] for k in indice]
				values_c = [values[0]*values[1],
										values[1]*values[3],
										values[0]*values[2],
										values[1]*values[3]]
				return tf.stack(values_c)

		return tf.map_fn(update_cross,com_label)


def tf_text_bboxes_encode(cord,
						 anchors, num,
						 gamma = 1.5,
						 dtype=tf.float32,
						 scope='text_bboxes_encode'):
		with tf.name_scope(scope):
				target_labels = []
				target_localizations = []
				target_links = []
				for i, anchors_layer in enumerate(anchors):
						with tf.name_scope('bboxes_encode_block_%i' % i):
								t_loc, t_label = \
										tf_text_bboxes_encode_layer(cord,anchors_layer, num, gamma,dtype)
								target_localizations.append(t_loc)
								target_labels.append(t_label)

				in_layer_link = in_layer_links_encode(target_labels[0])
				cross_layer_link = tf.ones((64*64,4), dtype = tf.int32)
				target_links.append(tf.concat([in_layer_link, cross_layer_link], -1))
				for i in range(1, 6):
					in_layer_link = in_layer_links_encode(target_labels[i])
					cross_layer_link = cross_layer_links_encode(target_labels[i-1], target_labels[i])	
					target_links.append(tf.concat([in_layer_link, cross_layer_link], -1))
				return target_localizations, target_labels, target_links


def textbox_anchor_one_layer(img_shape,
							 feat_size,
							 gamma = 1.5,
							 offset = 0.5,
							 dtype=np.float32):

		y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]] + offset
		y = y / feat_size[0] 
		x = y / feat_size[1]
		height = gamma * img_shape[0] / feat_size[0]
		return y, x , height, height



## produce anchor for all layers
def textbox_anchor_all_layers(img_shape,
													 layers_shape,
													 gamma=1.5,
													 offset=0.5,
													 dtype=np.float32):
		"""
		Compute anchor boxes for all feature layers.
		"""
		layers_anchors = []
		for i, s in enumerate(layers_shape):
				anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
																								 gamma=gamma,
																								 offset=offset, dtype=dtype)
				layers_anchors.append(anchor_bboxes)
		return layers_anchors



