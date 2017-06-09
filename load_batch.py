# *_* coding:utf-8 *_*

"""
This script produce a batch trainig 
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf 
from datasets import sythtextprovider
import tf_utils
from processing import txt_preprocessing
slim = tf.contrib.slim


def get_batch(dataset_dir,
			  num_readers,
			  batch_size,
			  out_shape,
			  net,
			  anchors,
			  FLAGS,
			  file_pattern = '*.tfrecord',
			  is_training = True,
			  shuffe = False):
	
	dataset = sythtextprovider.get_datasets(dataset_dir,file_pattern = file_pattern)

	provider = slim.dataset_data_provider.DatasetDataProvider(
				dataset,
				num_readers=num_readers,
				common_queue_capacity=20 * batch_size,
				common_queue_min=10 * batch_size,
				shuffle=shuffe)
	
	[image, shape, glabels, gbboxes, corx, cory] = provider.get(['image', 'shape',
															'object/label',
															'object/bbox',
															'object/corx',
															'object/cory'])
	corx = tf.expand_dims(corx,-1)
	cory = tf.expand_dims(cory,-1)

	cord = tf.concat([corx, cory],-1)

	if is_training:
		image, glabels, gbboxes, cord, num = \
		txt_preprocessing.preprocess_image(image,  glabels,gbboxes, cord,
										   out_shape,is_training=is_training)

		glocalisations, glabels, glinks = \
			net.bboxes_encode(cord, anchors ,num)

		batch_shape = [1] + [len(anchors)] * 3


		r = tf.train.batch(
			tf_utils.reshape_list([image, glocalisations, glabels, glinks]),
			batch_size=batch_size,
			num_threads=FLAGS.num_preprocessing_threads,
			capacity=5 * batch_size,
			)

		b_image, b_glocalisations, b_glabels, b_glinks= \
			tf_utils.reshape_list(r, batch_shape)

		return b_image, b_glocalisations, b_glabels, b_glinks
	else:
		image, labels, bboxes, cord, num = \
		txt_preprocessing.preprocess_image(image,  glabels,gbboxes, cord,
										out_shape,is_training=is_training)

		glocalisations, glabels, glinks = \
			net.bboxes_encode(cord, anchors ,num)

		batch_shape = [1]*3 + [len(anchors)] * 3


		r = tf.train.batch(
			tf_utils.reshape_list([image, labels, cord, glocalisations, glabels, glinks]),
			batch_size=batch_size,
			num_threads=FLAGS.num_preprocessing_threads,
			capacity=5 * batch_size,
			)

		b_image, b_labels, b_cord, b_glocalisations, b_glabels, b_glinks= \
			tf_utils.reshape_list(r, batch_shape)

		return b_image, b_labels, b_cord, b_glocalisations, b_glabels, b_glinks

