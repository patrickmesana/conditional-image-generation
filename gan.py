#
# Keras GAN Implementation
# Forked from: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
#
import os, random
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input, merge
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
from keras.models import Model
from keras.utils import np_utils
from tqdm import tqdm
import readsample as rs
import display

x_train_input = rs.read_images_from_pkl('training_input.pkl')
x_train_target = rs.read_images_from_pkl('training_target_full.pkl')
x_train_input = x_train_input.astype('float32') / 255.
x_train_target = x_train_target.astype('float32') / 255.
x_train_input = x_train_input.reshape((len(x_train_input), 64, 64, 3))
x_train_target = x_train_target.reshape((len(x_train_target), 64, 64, 3))

X_train = x_train_target
X_test = x_train_target

print np.min(X_train), np.max(X_train)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


shp = X_train.shape[1:]
dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

# Build Generative model ...
nch = 200
g_input = Input(shape=[100])
H = Dense(32 * 32 * nch, kernel_initializer='glorot_normal')(g_input)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Reshape([32, 32, nch])(H)
H = UpSampling2D(size=(2, 2))(H)
H = Convolution2D(nch / 2, (3, 3), padding="same", kernel_initializer='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(nch / 4, (3, 3), padding="same", kernel_initializer='glorot_uniform')(H)
H = BatchNormalization()(H)
H = Activation('relu')(H)
H = Convolution2D(3, (1, 1), padding="same", kernel_initializer='glorot_uniform')(H)
g_V = Activation('sigmoid')(H)
generator = Model(g_input, g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()

# Build Discriminative model ...
d_input = Input(shape=shp)
H = Convolution2D(256, (5, 5), padding="same", strides=(2, 2), activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, (5, 5), padding="same", strides=(2, 2), activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
discriminator.summary()

# the weights are not updated, need to compile again to take into account
# Freeze weights in the discriminator for stacked training
def make_disciminator_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
    net.compile(loss='categorical_crossentropy', optimizer=Adam())



make_disciminator_trainable(discriminator, False)

# Build stacked GAN model
gan_input = Input(shape=[100])
H = generator(gan_input)
gan_V = discriminator(H) # set of layers?
GAN = Model(gan_input, gan_V)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()


def plot_loss(losses):
    #        display.clear_output(wait=True)
    #        display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()

import PIL.Image as Image
def plot_gen(n_ex=5):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)
    for i in range(0, n_ex):
        Image.fromarray(generated_images[i]).show()

ntrain = 10000
# we geneate 10000 indexes of the size of mnist training set.
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
# instead of taking a range we take 100000 random indexes
XT = X_train[trainidx, :, :, :]

# Pre-train the discriminator network ... (better weight initialization)
# noise as input of generative
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100])
# transform the noise[100] into an image
generated_images = generator.predict(noise_gen)
# create a new set for discriminator containing generated images and real images
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
# 1 if real, 0 if fake image
y = np.zeros([2 * n, 2])
y[:n, 1] = 1
y[n:, 0] = 1

# train discriminator for 1 epoch
make_disciminator_trainable(discriminator, True)
discriminator.fit(X, y, epochs=1, batch_size=128)

# predict fake or real for the new training set created above
y_hat = discriminator.predict(X)

# Measure accuracy of pre-trained discriminator network
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx
n_tot = y.shape[0]
n_rig = (diff == 0).sum()
acc = n_rig * 100.0 / n_tot
print "Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot)

# set up loss storage vector
losses = {"d": [], "g": []}


# Set up our main training loop
def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):

        # Make generative images
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)

        # Train discriminator on generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_disciminator_trainable(discriminator, True)
        d_loss = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)

        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        make_disciminator_trainable(discriminator, False)
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)

        # Updates plots
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()


# Train for 6000 epochs at original learning rates
train_for_n(nb_epoch=6000, plt_frq=500, BATCH_SIZE=32)

# Train for 2000 epochs at reduced learning rates
# opt.lr.set_value(1e-5)
# dopt.lr.set_value(1e-4)
# train_for_n(nb_epoch=2000, plt_frq=500, BATCH_SIZE=32)
#
# # Train for 2000 epochs at reduced learning rates
# opt.lr.set_value(1e-6)
# dopt.lr.set_value(1e-5)
# train_for_n(nb_epoch=2000, plt_frq=500, BATCH_SIZE=32)

# Plot the final loss curves
plot_loss(losses)

# Plot some generated images from our GAN
plot_gen(25, (5, 5), (12, 12))