
import tensorflow as tf
import numpy as np
import math
import tf_extended as tfe





# =========================================================================== #
# TensorFlow implementation of Text Boxes encoding / decoding.
# =========================================================================== #

def tf_text_bboxes_encode_layer(bboxes,
                               anchors_layer, num,
                               match_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    
    pass



def tf_text_bboxes_encode(bboxes,
                         anchors, num,
                         match_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='text_bboxes_encode'):
    pass


def textbox_anchor_one_layer(img_shape,
                             feat_size,
                             gamma = 1.5,
                             offset = 0.5,
                             dtype=np.float32):

    y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]]
    y = (y.astype(dtype) + offset) / feat_size[0] 
    x = (x.astype(dtype) + offset) / feat_size[1]
    y_out = np.expand_dims(y, axis=-1)
    x_out = np.expand_dims(x, axis=-1)

    return y_out, x_out, gamma, gamma



## produce anchor for all layers
def textbox_achor_all_layers(img_shape,
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



