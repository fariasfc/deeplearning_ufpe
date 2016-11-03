import argparse

import numpy as np
import time
import pandas as pd
import theano
from os.path import isfile
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theano.config.mode='FAST_RUN'#'DEBUG_MODE'#'FAST_COMPILE'#'DEBUG_MODE'#
# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'
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

# import tensorflow as tf
# import tensorflow
# from tensorflow.python.ops import control_flow_ops
# tensorflow.python.control_flow_ops = control_flow_ops
SEED = 1
np.random.seed(SEED)

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
                 nesterov=False, threshold=0, algorithm=None, **kwargs):
        super(SSGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.threshold = threshold
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)
        self.inital_decay = decay
        self.algorithm = algorithm

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
            if self.algorithm == 'droplowests_probs':
                print('droping lowests with probability!')
                r = K.random_uniform(g.shape, seed=SEED)
                normalized = T.abs_(g)/T.max(g)
                mask = r < normalized
                new_g = g * mask
            elif self.algorithm == 'drophighests_probs':
                print('droping highests with probability!')
                r = K.random_uniform(g.shape, seed=SEED)
                normalized = T.abs_(g)/T.max(g)
                mask = r > normalized
                new_g = g * mask
            elif self.algorithm == 'dropgrads':
                print("dropgrads!")
                mask = K.random_binomial(g.shape, p=1-threshold, seed=SEED)
                new_g = mask*g
            elif self.algorithm == 'droplowests':
                print("droplowests!")
                g_abs = K.abs(g)
                max_g = K.max(g_abs)
                min_g = K.min(g_abs)
                threshold_values = threshold * (max_g - min_g) + min_g
                mask = g_abs > threshold_values
                new_g = g * mask
            elif self.algorithm == 'drophighests':
                print("drophighests!")
                g_abs = K.abs(g)
                max_g = K.max(g_abs)
                min_g = K.min(g_abs)
                threshold_values = threshold * (max_g - min_g) + min_g
                mask = g_abs < threshold_values
                new_g = g * mask
            elif self.algorithm == 'dropout':
                print("dropout!")
            else:
                raise Exception('invalid algorithm')

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



def create_model(shape_inputs, nb_classes, kernel_size, pool_size, strides, algorithm, threshold):
    weights_file = dataset_name+'_weights.h5'
    inputs = Input(shape=shape_inputs, name='inputs')
    predictions = None
    if dataset_name == 'cifar10':
        if algorithm == 'dropout':
            h = Dropout(0.9, name='drop_inputs')(inputs)
            h = Convolution2D(96, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(h)
        else:
            h = Convolution2D(96, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(inputs)

        h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp1')(h)
        if algorithm == 'dropout':
            h = Dropout(0.75, name='drop1')(h)

        h = Convolution2D(128, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv2')(h)
        h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp2')(h)
        if algorithm == 'dropout':
            h = Dropout(0.75, name='drop2')(h)

        h = Convolution2D(256, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv3')(h)
        h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp3')(h)
        if algorithm == 'dropout':
            h = Dropout(0.5, name='drop3')(h)

        # h = Convolution2D(64, 3, 3, border_mode='same',activation='relu')(inputs)
        # h = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(h)
        # h = MaxPooling2D(pool_size=(2, 2))(h)
        # h = Dropout(0.25)(h)

        h = Flatten()(h)
        h = Dense(2048, activation='relu', name='dense1')(h)
        if algorithm == 'dropout':
            h = Dropout(0.5, name='drop4')(h)
        h = Dense(2048, activation='relu', name='dense2')(h)
        if algorithm == 'dropout':
            h = Dropout(0.5, name='drop_outputs')(h)
        predictions = Dense(nb_classes, activation='softmax', name='outputs')(h)
    elif dataset_name == 'mnist':
        h = Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv1')(inputs)
        h = Convolution2D(32, kernel_size[0], kernel_size[1], border_mode='same', activation='relu', name='conv2')(h)
        h = MaxPooling2D(pool_size=pool_size, strides=strides, name='maxp2')(h)
        if algorithm == 'dropout':
            h = Dropout(0.25, name='drop1')(h)

        h = Flatten()(h)
        h = Dense(128, activation='relu', name='dense1')(h)
        if algorithm == 'dropout':
            h = Dropout(0.5, name='drop2')(h)
        predictions = Dense(nb_classes, activation='softmax', name='outputs')(h)


    # this creates a model that includes
    # the Input layer and three Dense layers
    model = Model(input=inputs, output=predictions)

    # let's train the model using SGD + momentum (how original).
    if (algorithm == 'dropout'):
        opt = SGD(lr=0.01, decay=1e-6, momentum=0, nesterov=False)
    else:
        opt = SSGD(lr=0.01, decay=1e-6, momentum=0, nesterov=False, threshold=threshold, algorithm=algorithm)

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
    parser.add_argument('--algorithm', type=str, default='dropout')
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()
    global dataset_name
    dataset_name = args.dataset
    print(args)
    # print('floatx=' + theano.config.floatX)

    use_ssgd = True
    nb_classes = 10
    nb_epoch = 200
    batch_size = 128
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
        
        filename = "{}-nb_epochs={}_{}_{}.csv".format(dataset_name, nb_epoch, args.algorithm, threshold)

        print('algorithm: {}    threshold: {}'.format(args.algorithm, threshold))

        model = create_model(X_train.shape[1:], nb_classes, kernel_size=kernel_size, pool_size=pool_size, strides=strides, algorithm=args.algorithm, threshold=threshold)

        callbacks = [
            history_callback,
            LambdaCallback(on_epoch_end=lambda epoch, logs: pd.DataFrame.from_dict(history_callback.history).to_csv(filename))
        ]
        if not data_augmentation:
            print('Not using data augmentation.')
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

        # df = pd.DataFrame.from_dict(history_callback.history)
        #
        # df.to_csv(filename)

if __name__ == "__main__":
    main()
