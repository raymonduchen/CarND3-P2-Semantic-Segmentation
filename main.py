import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'
# Hyperparameters
epochs = 41             # 11, 21, 31, 41
batch_size = 10          # 5
keep_prob_value = 0.5   
learning_rate_value = 0.0001  

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #----------- (transfer VGG model as encoder)
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)        # load graph from file
    graph = tf.get_default_graph()                              # grab the graph in graph variable

    input = graph.get_tensor_by_name(vgg_input_tensor_name)        # grab layer in graph by name
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input, keep_prob, layer_3, layer_4, layer_7

print("TEST_LOAD_VGG : -----")
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # 1x1 convolution [downsample]
    conv7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='conv7_1x1')
    conv4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='same', 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='conv4_1x1')
    conv3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='same', 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='conv3_1x1')

    # 1st deconvolution x2 [upsample]
    upsample1_x2 = tf.layers.conv2d_transpose(conv7_1x1, num_classes, 4, strides=(2, 2), padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='upsample1_x2')

    # Add 1st skip layer : upsample1_x2 + conv4_1x1
    skip1 = tf.add(upsample1_x2, conv4_1x1)

    # 2nd deconvolution x2 [upsample]
    upsample2_x2 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides=(2, 2), padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='upsample2_x2')

    # Add 2nd skip layer : upsample2_x2 + conv3_1x1
    skip2 = tf.add(upsample2_x2, conv3_1x1)
    
    # 3rd deconvolution x8 [upsample]
    upsample3_x8 = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                name='upsample3_x8')    

    return upsample3_x8

print("TEST_LAYERS : -----")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    #---
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
    #return None, None, None

print("TEST_OPTIMIZE : -----")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    #--
    for epoch_i in range(epochs):
        loss_data = []
        for image, label in get_batches_fn(batch_size):
            # Training
            _, loss = sess.run([train_op, cross_entropy_loss], 
                                feed_dict = {input_image: image, 
                                             correct_label: label,
                                             keep_prob: keep_prob_value,
                                             learning_rate: learning_rate_value
                                            }
                                )
            loss_data.append(loss)
        loss_mean = np.mean(loss_data)
        loss_var = np.var(loss_data)
        print('Epoch{:<3} - Mean:{} - Var:{} - Loss_data: {}'.format(epoch_i, loss_mean, loss_var, loss_data))

print("TEST_TRAIN_NN : -----")
tests.test_train_nn(train_nn)


def run():
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model [if there's no vgg model in data folder ...]
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        #---
        input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)

        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer()) 

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input, correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)
        
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
