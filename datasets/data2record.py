## create script that download datasets and transform into tf-record
## Assume the datasets is downloaded into following folders
## SythTexts datasets(41G)
## data/sythtext/*

import numpy as np 
import scipy.io as sio
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf
import re
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm

from PIL import Image

data_path = '../../TextBoxes-TensorFlow/data/sythtext/'
cellname = 'gt'
textname = 'txt'
imcell = 'imnames'
wordname = 'wordBB'
charname = 'charBB'
NUMoffolder = 200

## SythText datasets is too big to store in a record. 
## So Transform tfrecord according to dir name


def _convert_to_example(image_data, shape, charbb, label,imname):

	#print 'shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1])
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/height': int64_feature(shape[0]),
			'image/width': int64_feature(shape[1]),
			'image/channels': int64_feature(shape[2]),
			'image/shape': int64_feature(shape),
			'image/object/bbox/x0': float_feature(charbb[0,0,:].tolist()),
			'image/object/bbox/x1': float_feature(charbb[0,1,:].tolist()),
			'image/object/bbox/x2': float_feature(charbb[0,2,:].tolist()),
			'image/object/bbox/x3': float_feature(charbb[0,3,:].tolist()),
			'image/object/bbox/y0': float_feature(charbb[1,0,:].tolist()),
			'image/object/bbox/y1': float_feature(charbb[1,1,:].tolist()),
			'image/object/bbox/y2': float_feature(charbb[1,2,:].tolist()),
			'image/object/bbox/y3': float_feature(charbb[1,3,:].tolist()),
			'image/object/bbox/label': int64_feature(label),
			'image/format': bytes_feature('jpeg'),
			'image/encoded': bytes_feature(image_data),
			'image/name': bytes_feature(imname.tostring()),
			}))
	return example
	

def _processing_image(charbb, imname,coder):
	image_data = tf.gfile.GFile(data_path+imname, 'r').read()
	image = coder.decode_jpeg(image_data)
	shape = image.shape
	numofbox = charbb.shape[2]
	#in the order [x1,x2,x3,x4,y1,y2,y3,y4]
	label = [1 for i in range(numofbox)]
	shape = list(shape)
	return image_data, shape, charbb, label, imname


def run():
	labels = sio.loadmat('../../TextBoxes-TensorFlow/data/sythtext/'+'gt.mat')
	print labels.keys()
	texts = labels[textname]
	imnames = labels[imcell]
	wordBB = labels[wordname]
	charBB = labels[charname]
	coder = ImageCoder()
	for i in range(NUMoffolder):
		tf_filename = str(i+1) + '.tfrecord'
		tfrecord_writer = tf.python_io.TFRecordWriter('../data/sythtext/' + tf_filename)
		dir = i+1
		pattern = re.compile(r'^{}\/'.format(dir))
		print dir
		print pattern
		res =[k for k in range(imnames.shape[1]) if pattern.match(imnames[0,k][0]) != None ]
		print "The size of %s folder : %s" % (dir,len(res))
		# shuffle
		res = np.random.permutation(res)
		for j in res:
			charbb = charBB[0,j]
			imname = imnames[0,j][0]
			#print str(i) + imname
			image_data, shape, bbox, label ,imname= _processing_image(charbb, imname,coder)

			example = _convert_to_example(image_data, shape, bbox, label, imname)
			tfrecord_writer.write(example.SerializeToString())  
	print 'Transform to tfrecord finished'

if __name__ == '__main__':
	run()





