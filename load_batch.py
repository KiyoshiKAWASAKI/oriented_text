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
	
	[image, shape, glabels, gcorx, gcory] = provider.get(['image', 'shape',
											 'object/label',
											 'object/corx',
											 'object/cory'])



	if is_training:
		image, glabels, gbboxes,num = \
		txt_preprocessing.preprocess_image(image,  glabels,gbboxes, 
										out_shape,use_whiten=FLAGS.use_whiten,is_training=is_training)

		'''

		'''
		return b_image, b_glocalisations, b_gscores
	else:
		image, glabels, gbboxes,bbox_img, num = \
		txt_preprocessing.preprocess_image(image,  glabels,gbboxes, 
										out_shape,use_whiten=FLAGS.use_whiten,is_training=is_training)

		#TODO
		'''

		'''
		return image, glabels, gbboxes, g_bbox_img, glocalisations, gscores

