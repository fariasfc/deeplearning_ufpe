import argparse

import numpy as np
import time
import pandas as pd
from os.path import isfile
import os

from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import History, BaseLogger, LambdaCallback
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import cifar10, mnist
import keras.backend as K
from keras.utils import np_utils

from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Optimizer

if K.backend() == 'theano':
    import theano
    import theano.tensor as T
    theano.config.mode='FAST_RUN'#'DEBUG_MODE'#'FAST_COMPILE'#'DEBUG_MODE'#
    # theano.config.optimizer='fast_compile'
    # theano.config.exception_verbosity='high'
    # theano.config.compute_test_value = 'warn'
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    import tensorflow as tf
    import tensorflow
    from tensorflow.python.ops import control_flow_ops
    tensorflow.python.control_flow_ops = control_flow_ops

SEED = 1
BATCH_SIZE = 128
np.random.seed(SEED)

def get_prefix(args, filetype, threshold=None, index=None):
    if filetype == 'initial_weights':
        prefix = args.dataset +'_model=' + args.model + "_idx=" + str(index) + "_weights.h5"
    elif filetype in ['trained', 'csv', 'model']:
        prefix = dataset_name +  '_nbepochs=' + str(args.nb_epochs) + '_model=' + args.model + '_opt=' + args.optimizer + '_dropoutmethod=' + args.dropout_method + '_scale=' + str(args.scale) + '_threshold=' + str(threshold) + '_index=' + str(index)

        if args.dropout_method == 'dropout_decayed':
            droprates = 'drop_rates='
            for drop_rate in args.drop_rates:
                droprates += str(drop_rate) + '_'
            prefix = prefix + droprates

        if filetype == 'trained':
            prefix = prefix + '_trained_weights.h5'
        elif filetype == 'csv':

            prefix = prefix + '.csv'
        else:
            prefix = prefix + '.txt'
    else:
        raise Exception("filetype must be specified")

    print(prefix)

    return prefix

class DropoutModified(Layer):
    '''Applies Dropout to the input. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p, method, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        self.method = method
        super(DropoutModified, self).__init__(**kwargs)
        # self.stateful = True

    def build(self, input_shape):
        print(input_shape)
        # self.drop_ages = K.ones((BATCH_SIZE,) + input_shape[1:], name='drop_ages')
        # self.drop_ages = K.random_binomial(input_shape[1:], p=0.5, seed=1)

        # self.drop_ages = tf.get_variable('drop_ages', dtype=np.float32, initializer=tf.random_uniform(input_shape[1:], 0, 1, dtype=np.int32, seed=SEED))
        self.drop_ages = K.ones(input_shape[1:], name='drop_ages')
        self.drop_ages = self.drop_ages * K.random_uniform(input_shape[1:], low=0.0, high=10) * K.random_binomial(input_shape[1:], p=0.5, seed=1)
        # self.drop_ages[:,:,:]=1
        print("created drop_ages tensor")

    def _get_noise_shape(self, x):
        return None

    def call(self, x, mask=None):
        # Original:
        # if 0. < self.p < 1.:
        #     noise_shape = self._get_noise_shape(x)
        #     x = K.in_train_phase(K.dropout(x, self.p, noise_shape), x)
        # return x

        print('type of x: {}'.format(type(x)))

        # for k in dir(self):
        #     print("{}={}".format(k, getattr(self,k)))
        # self.drop_ages = K.zeros(K.int_shape(x))
        if 0. < self.p < 1.:
            print("using " + self.method)
            if self.method == 'dropout_highests_probs':
                r = K.random_uniform(x.shape, seed=SEED)
                normalized = K.abs(x) / K.max(x)
                mask = r > normalized
                new_x = x * mask
                x = K.in_train_phase(new_x, x)
                return x
            elif self.method == 'dropout_lowests_probs':
                print('droping lowests with probability!')
                r = K.random_uniform(x.shape, seed=SEED)
                normalized = K.abs(x)/K.max(x)
                mask = r < normalized
                new_x = x * mask
                x = K.in_train_phase(new_x, x)
                return x
            elif self.method == 'dropout_oldests':
                retain_prob = 1. - self.p
                shape = K.shape(x)
                random_tensor = K.random_uniform(shape[1:], dtype=x.dtype, seed=1)
                # random_binomial = K.random_binomial(shape[1:], p=retain_prob, dtype=x.dtype, seed=1)
                #
                drop_max = K.max(self.drop_ages)
                # if drop_max == 0:
                #     normalized_ages = self.drop_ages
                # else:
                #     normalized_ages = self.drop_ages/drop_max
                #
                # # older parameters tend to be selected... 1 retains, 0 drops
                # mask_lesser = K.lesser(random_tensor, normalized_ages)
                self.mask_lesser = K.lesser_equal(random_tensor, self.drop_ages/drop_max)
                drop = K.cast(self.mask_lesser, K.floatx())
                # self.casted_mask = K.cast(self.mask_lesser, K.floatx())
                # expanded_dropout_matrix = K.repeat(self.casted_mask, BATCH_SIZE)

                # self.drop = self.casted_mask * random_binomial
                # self.drop = self.casted_mask
                #
                # # zeroing the dropped + non_dropped
                # # dropped = drop*drop_ages + drop
                new_ages = (1+self.drop_ages) * drop
                #
                new_x = x * drop
                # drop_mean = K.mean(self.drop)
                # drop_mean = K.print_tensor(drop_mean, 'drop_mean: ')
                # if drop_mean != 0:
                #     new_x /= K.mean(self.drop)
                # # print(new_x)
                # new_x = K.print_tensor(new_x, 'this is new_x: ')

                self.updates = [self.drop_ages, new_ages]
                #
                x = K.in_train_phase(new_x, x)
                # x = K.in_train_phase(x*random_binomial, x)
                return x
            # Original:
            #     noise_shape = self._get_noise_shape(x)
            #     x = K.in_train_phase(K.dropout(x, self.p, noise_shape), x)
            else:
                raise Exception('dropout method unknown' + self.method)


        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(DropoutModified, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DropoutDecayed(Layer):
    '''Applies Dropout to the input. Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    '''
    def __init__(self, p_start, p_end, nb_iterations, **kwargs):
        self.p_start = p_start
        self.p_end = p_end

        self.decay = (p_end - p_start) / nb_iterations

        self.supports_masking = True
        self.iterations = K.variable(0.)
        self.p = K.variable(p_start)

        if 0. < p_start < 1. and 0. < p_end < 1.:
            self.uses_learning_phase = True
        self.updates = []

        super(DropoutDecayed, self).__init__(**kwargs)
        # self.stateful = True

    def build(self, input_shape):
        print(input_shape)
        # self.drop_ages = K.ones((BATCH_SIZE,) + input_shape[1:], name='drop_ages')
        # self.drop_ages = K.random_binomial(input_shape[1:], p=0.5, seed=1)

        # self.drop_ages = tf.get_variable('drop_ages', dtype=np.float32, initializer=tf.random_uniform(input_shape[1:], 0, 1, dtype=np.int32, seed=SEED))
        # self.drop_ages = K.ones(input_shape[1:], name='drop_ages')
        # self.drop_ages = self.drop_ages * K.random_uniform(input_shape[1:], low=0.0, high=10) * K.random_binomial(input_shape[1:], p=0.5, seed=1)
        # self.drop_ages[:,:,:]=1

        print("built DropoutDecayed")

    def _get_noise_shape(self, x):
        return None

    def call(self, x, mask=None):
        # Original:
        # if 0. < self.p < 1.:
        #     noise_shape = self._get_noise_shape(x)
        #     x = K.in_train_phase(K.dropout(x, self.p, noise_shape), x)
        # return x

        print('type of x: {}'.format(type(x)))

        # for k in dir(self):
        #     print("{}={}".format(k, getattr(self,k)))
        # self.drop_ages = K.zeros(K.int_shape(x))

        p = self.p

        p += self.decay*self.iterations
        self.updates.append(K.update_add(self.iterations, 1))


        noise_shape = self._get_noise_shape(x)


        rng = RandomStreams(seed=SEED)
        retain_prob = 1. - p


        random_tensor = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)

        x_new = x * random_tensor
        x_new /= retain_prob

        x = K.in_train_phase(x_new, x)

        return x

    def get_config(self):
        config = {'p_start': self.p_start, 'p_end': self.p_end, 'decay': self.decay}
        base_config = super(DropoutDecayed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SSGD(Optimizer):
    '''Stochastic Selective gradient descent, with support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, threshold=0, optimizer=None, scale=False, **kwargs):
        super(SSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.threshold = threshold
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)
        self.inital_decay = decay
        self.optimizer = optimizer
        self.scale = scale

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)

        self.updates = []

        lr = self.lr
        threshold = self.threshold
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            # threshold *= (1. / (1. + self.decay * self.iterations))

            self.updates.append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            # sorted_gradients = T.argsort(g)
            # nb_params = T.shape(sorted_gradients)
            # cutoff_value = sorted_gradients[int(threshold*nb_params)]
            # new_g = g * g[g > cutoff_value]
            if self.optimizer == 'droplowests_probs':
                print('droping lowests with probability!')
                r = K.random_uniform(g.shape, seed=SEED)
                normalized = T.abs_(g)/T.max(g)
                mask = r < normalized
                new_g = g * mask
            elif self.optimizer == 'drophighests_probs':
                print('droping highests with probability!')
                r = K.random_uniform(g.shape, seed=SEED)
                normalized = T.abs_(g)/T.max(g)
                mask = r > normalized
                new_g = g * mask
            elif self.optimizer == 'dropgrads':
                print("dropgrads!")
                mask = K.random_binomial(g.shape, p=1-threshold, seed=SEED)
                new_g = mask*g
            elif self.optimizer == 'droplowests':
                print("droplowests!")
                g_abs = K.abs(g)
                max_g = K.max(g_abs)
                min_g = K.min(g_abs)
                threshold_values = threshold * (max_g - min_g) + min_g
                mask = g_abs > threshold_values
                new_g = g * mask
            elif self.optimizer == 'drophighests':
                print("drophighests!")
                g_abs = K.abs(g)
                max_g = K.max(g_abs)
                min_g = K.min(g_abs)
                threshold_values = threshold * (max_g - min_g) + min_g
                mask = g_abs < threshold_values
                new_g = g * mask
            else:
                raise Exception('invalid optimizer')

            if self.scale:
                new_g = new_g/(1-threshold)

            v = self.momentum * m - lr * new_g  # velocity
            self.updates.append(K.update(m, v))


            if self.nesterov:
                new_p = p + self.momentum * v - lr * new_g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def create_model(shape_inputs, nb_classes, kernel_size, pool_size, strides, threshold, index, args):
    weights_file = get_prefix(args, 'initial_weights', None, index)
    dropout_method = args.dropout_method
    optimizer = args.optimizer
    scale = args.scale
    model = args.model
    inputs = Input(shape=shape_inputs, name='inputs')
    predictions = None
    # if model == '96x128x256x2048x2048':
    #     if dropout_method == 'original':
    #         h = Dropout(0.9, name='drop_inputs')(inputs)
    #         h = Convolution2D(96, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(h)
    #     else:
    #         h = Convolution2D(96, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(inputs)
    #
    #     h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp1')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.75, name='drop1')(h)
    #
    #     h = Convolution2D(128, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv2')(h)
    #     h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp2')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.75, name='drop2')(h)
    #
    #     h = Convolution2D(256, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv3')(h)
    #     h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp3')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.5, name='drop3')(h)
    #
    #     # h = Convolution2D(64, 3, 3, border_mode='same',activation='relu')(inputs)
    #     # h = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(h)
    #     # h = MaxPooling2D(pool_size=(2, 2))(h)
    #     # h = Dropout(0.25)(h)
    #
    #     h = Flatten()(h)
    #     h = Dense(2048, activation='relu', name='dense1')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.5, name='drop4')(h)
    #     h = Dense(2048, activation='relu', name='dense2')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.5, name='drop_outputs')(h)
    #     predictions = Dense(nb_classes, activation='softmax', name='outputs')(h)
    if model == '32x32x128':
        h = Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(inputs)
        h = Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv2')(h)
        h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp2')(h)
        drop_rate = 0.25
        if dropout_method == 'original':
            h = Dropout(drop_rate, name='drop1')(h)
        elif 'dropout' in dropout_method:
            if dropout_method == 'dropout_decayed':
                h = DropoutDecayed(args.drop_rates[0], args.drop_rates[1], args.nb_epochs*50000/128)(h)
            else:
                h = DropoutModified(drop_rate, name='drop_modified1' + dropout_method, method=dropout_method)(h)


        h = Flatten()(h)
        h = Dense(128, activation='relu', name='dense1')(h)
        drop_rate = 0.5
        if dropout_method == 'original':
            h = Dropout(drop_rate, name='drop2')(h)
        elif 'dropout' in dropout_method:
            if dropout_method == 'dropout_decayed':
                h = DropoutDecayed(args.drop_rates[2], args.drop_rates[3], args.nb_epochs * 50000/128)(h)
            else:
                h = DropoutModified(drop_rate, name='drop_modified2' + dropout_method, method=dropout_method)(h)
        predictions = Dense(nb_classes, activation='softmax', name='outputs')(h)
    # elif model == '32x64x128x1024x512':
    #     # Create the model
    #
    #     h = Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(inputs)
    #     if dropout_method == 'original':
    #         h = Dropout(0.2)(h)
    #     h = Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv2')(h)
    #     h = MaxPooling2D(pool_size=pool_size)(h)
    #
    #     h = Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv3')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.2)(h)
    #     h = Convolution2D(64, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv4')(h)
    #     h = MaxPooling2D(pool_size=pool_size)(h)
    #
    #     h = Convolution2D(128, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv5')(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.2)(h)
    #     h = Convolution2D(128, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv6')(h)
    #     h = MaxPooling2D(pool_size=pool_size)(h)
    #
    #     h = Flatten()(h)
    #     if dropout_method == 'original':
    #         h = Dropout(0.2)(h)
    #
    #     h = Dense(1024, activation='relu')(h)
    #
    #     if dropout_method == 'original':
    #         h = Dropout(0.2)(h)
    #
    #     h = Dense(512, activation='relu')(h)
    #
    #     if dropout_method == 'original':
    #         h = Dropout(0.2)(h)
    #
    #     predictions = Dense(nb_classes, activation='softmax', name='outputs')(h)
    else:
        raise Exception('Unknown model')



    # this creates a model that includes
    # the Input layer and three Dense layers
    model = Model(input=inputs, output=predictions)

    # let's train the model using SGD + momentum (how original).
    if (optimizer == 'sgd'):
        opt = SGD(lr=0.01, decay=1e-6, momentum=0, nesterov=False)
    else:
        opt = SSGD(lr=0.01, decay=1e-6, momentum=0, nesterov=False, threshold=threshold, optimizer=optimizer, scale=scale)

    print('Compiling model...')
    start_compile = time.time()
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    end_compile = time.time()
    print('Compiled in {}'.format(end_compile - start_compile))

    # if algorithm:
    #     model_yaml = model.to_yaml()
    #     filename = dataset_name+'sorted_ssgd_model.txt'
    # else:
    #     model_yaml = model.to_yaml()
    #     filename = dataset_name+'sgd_model.txt'

    # with open(filename, "w") as text_file:
    #     print("{}".format(model_yaml), file=text_file)

    if args.transfer_learning:
        print("Transfer Learning: {}".format(weights_file))
        trained_weights_file = get_prefix(args, 'trained', None, index)
        assert isfile(trained_weights_file), "Transfer Learning activated, but file {} not found".format(trained_weights_file)
        model.load_weights(trained_weights_file, by_name=True)
        print("Loaded weights from: {}".format(trained_weights_file))
    else:
        if isfile(weights_file):
            print("Loading Weights File: {} ...".format(weights_file))
            model.load_weights(weights_file, by_name=True)
        else:
            print("Saving Weights File: {} ...".format(weights_file))
            model.save_weights(weights_file)

    return model

dataset_name='mnist'

def main():
    print("starting!")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--thresholds', type=float, metavar='T', nargs='+', default=[-1])
    parser.add_argument('--drop_rates', type=float, metavar='T', nargs='+', default=[-1])
    parser.add_argument('--nb_epochs', type=int, metavar='E', default=200)
    parser.add_argument('--nb_runs', type=int, metavar='E', default=5)
    # parser.add_argument('--algorithm', type=str, default='nothing')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--scale', action='store_true', default=False)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--transfer_learning', action='store_true', default=False)
    # parser.add_argument('--usedropout', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='32x32x128')
    parser.add_argument('--dropout_method', type=str, default='no')

    args = parser.parse_args()
    global dataset_name
    dataset_name = args.dataset
    print(args)
    nb_classes = 10
    nb_epoch = args.nb_epochs
    batch_size = BATCH_SIZE
    data_augmentation = False
    print(dataset_name)
    if dataset_name == 'cifar10':
        # input image dimensions
        img_rows, img_cols = 32, 32
        # the CIFAR10 images are RGB
        img_channels = 3
        kernel_size = (5, 5)
        pool_size = (3, 3)
        strides = (2, 2)
    elif dataset_name == 'mnist':
        # input image dimensions
        img_rows, img_cols = 28, 28
        # the CIFAR10 images are RGB
        img_channels = 1
        kernel_size = (3, 3)
        pool_size = (2, 2)
        strides = (1, 1)
    else:
        raise Exception("Invalid Dataset")

    # the data, shuffled and split between train and test sets
    if dataset_name == 'cifar10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # max_idx_train = 5000
    # max_idx_test = 1000
    # X_train = X_train[:max_idx_train, :, :, :]
    # X_test = X_test[:max_idx_test, :, :, :]
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Y_train = Y_train[:max_idx_train]
    # Y_test = Y_test[:max_idx_test]

    start = time.time()
    print(X_train.shape)
    X_train = X_train.astype(np.float32)#theano.config.floatX)
    X_test = X_test.astype(np.float32)#theano.config.floatX)
    X_train /= 255
    X_test /= 255
    print("Normalized in {}".format(time.time()-start))
    history_callback = History()

    # import tensorflow as tf
    # sess = tf.InteractiveSession()
    # K.set_session(sess)
    # with tf.device('/gpu:0'):

    for threshold in args.thresholds:
        for index in range(args.nb_runs):
            print("threshold: {}".format(threshold))
            model = create_model(X_train.shape[1:], nb_classes, kernel_size=kernel_size, pool_size=pool_size, strides=strides, threshold=threshold, index=index, args=args)

            filename = get_prefix(args, 'model', threshold, index=index)#"_".join([dataset_name, args.optimizer, args.dropout_method, 'scale='+str(args.scale), str(threshold),'model', args.model])+'.txt'
            for l in model.layers:
                print(l.get_config())

            print(model.summary())
            print("writing model to " + filename)
            with open(filename, "w") as text_file:
                text_file.write(model.to_json())

                # print("{}".format(model.to_json()), file=text_file)

            filename = get_prefix(args, 'csv', threshold, index=index) #"{}-nb_epochs={}_{}_{}_{}_{}_{}.csv".format(dataset_name, nb_epoch, args.dropout_method, 'scale='+str(args.scale), threshold, args.optimizer, args.model)
            print('saving csv in ' + filename)
            callbacks = [
                history_callback,
                LambdaCallback(on_epoch_end=lambda epoch, logs: pd.DataFrame.from_dict(history_callback.history).to_csv(filename))
            ]
            if not args.augmentation:
                print('Not using data augmentation.')

                # ##### view values: ####
                # epochs = 25
                # print('Training')
                # for i in range(epochs):
                #     print('Epoch', i, '/', epochs)
                #     # nb_batches = 2
                #
                #     # model.fit(X_train[i*nb_batches * batch_size:(i+1)*nb_batches * batch_size],
                #     #           Y_train[i*nb_batches * batch_size:(i+1)*nb_batches * batch_size],
                #     #           batch_size=batch_size,
                #     #           verbose=1,
                #     #           nb_epoch=1,
                #     #           shuffle=False)
                #     l = model.layers[4]
                #     print("p={} decay={} iterations={} current_p={}".format(K.get_value(l.p), l.decay, K.get_value(l.iterations), K.get_value(l.p) + l.decay * K.get_value(l.iterations)))
                #
                #     model.fit(X_train,
                #               Y_train,
                #               batch_size=batch_size,
                #               verbose=1,
                #               nb_epoch=1,
                #               shuffle=False)
                #     # print('drop = {}'.format(K.get_value(model.layers[4].drop_ages)))
                #
                #     # for layer in model.layers:
                #     #     if 'DropoutModified' in str(layer):
                #     #         # print('mask_lesser = {}'.format(K.get_value(layer.mask_lesser)))
                #     #         # print('casted_mask = {}'.format(K.get_value(layer.casted_mask)))
                #     #         print('drop = {}'.format(K.get_value(layer.drop)))
                #
                #     # output of the first batch value of the batch after the first fit().
                #     # first_batch_element = np.expand_dims(cos[0], axis=1)  # (1, 1) to (1, 1, 1)
                #     # print('output = {}'.format(get_LSTM_output([first_batch_element])[0].flatten()))
                #
                #     # model.reset_states()
                #     ### END view values ###

                model.fit(X_train, Y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_test, Y_test),
                          shuffle=True,
                          callbacks=callbacks, verbose=args.verbose)
                print(callbacks)
            else:
                print('Using real-time data augmentation.')

                # this will do preprocessing and realtime data augmentation
                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images

                # compute quantities required for featurewise normalization
                # (std, mean, and principal components if ZCA whitening is applied)
                datagen.fit(X_train)


                # fit the model on the batches generated by datagen.flow()
                model.fit_generator(datagen.flow(X_train, Y_train,
                                    batch_size=batch_size),
                                    samples_per_epoch=X_train.shape[0],
                                    nb_epoch=nb_epoch,
                                    validation_data=(X_test, Y_test),
                                    callbacks=callbacks,)

            weights_file = get_prefix(args, 'trained', threshold, index)#dataset_name + '_' + args.optimizer + '_' + args.dropout_method + '_'+ str(threshold) + '_'+ args.optimizer + '_' +'_trained_weights' + '_' + str(nb_epoch) + '_' + args.model + '.h5'
            print("Saving Weights File: {} ...".format(weights_file))
            model.save_weights(weights_file)

            # df = pd.DataFrame.from_dict(history_callback.history)
            #
            # df.to_csv(filename)

if __name__ == "__main__":
    main()
