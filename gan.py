#
# Keras GAN Implementation
# Forked from: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
#
import os, random, shutil
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input, merge
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2D
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
from matplotlib.backends.backend_pdf import PdfPages
import copy
import PIL.Image as Image
from keras import losses
import time

if os.path.exists('gan_decoded.pkl'):
    os.remove('gan_decoded.pkl')
if os.path.exists('gan_summary.pdf'):
    os.remove('gan_summary.pdf')
logs_enabled = True
if logs_enabled:
    # removes logs if necessary and create it
    dir = './logs'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    loss_logs = open('./logs/loss_logs', 'w')


def show_denormalized(image):
    Image.fromarray((image * 255).astype('uint8')).show()


x_train_input = rs.read_images_from_pkl('training_input.pkl')
x_train_target = rs.read_images_from_pkl('training_target_full.pkl')
x_test_input_unchanged = rs.read_images_from_pkl('validation_input.pkl')
x_test_target = rs.read_images_from_pkl('validation_target_full.pkl')
x_train_input = x_train_input.astype('float32') / 255.
x_train_target = x_train_target.astype('float32') / 255.
x_test_input = x_test_input_unchanged.astype('float32') / 255.
# x_test_target = x_test_target.astype('float32') / 255.
x_train_input = x_train_input.reshape((len(x_train_input), 64, 64, 3))
# x_train_target = x_train_target.reshape((len(x_train_target), 64, 64, 3))
x_test_input = x_test_input.reshape((len(x_test_input), 64, 64, 3))
# x_test_target = x_test_target.reshape((len(x_test_target), 64, 64, 3))


X_train = x_train_target
X_test = x_train_target

print np.min(X_train), np.max(X_train)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# the weights are not updated, need to compile again to take into account
# Freeze weights in the discriminator for stacked training
def make_disciminator_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
    net.compile(loss='categorical_crossentropy', optimizer=Adam())

shp = X_train.shape[1:]
dropout_rate = 0.25

input_img = Input(shape=(64, 64, 3))

x = Conv2D(16, (3, 3), padding='same')(input_img)
x = Activation('relu')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = Activation('relu')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# this model maps an input to its reconstruction
generator = Model(input_img, decoded)
generator.load_weights('./g_pre_weights.hdf5')
generator.compile(optimizer=Adam(0.00001), loss='binary_crossentropy')
generator.summary()

# Build Discriminative model ...
d_input = Input(shape=shp)
H = Convolution2D(16, (3, 3), padding="same")(d_input)
H = MaxPooling2D((2, 2))(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(16, (3, 3), padding="same")(H)
H = MaxPooling2D((2, 2))(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(32, (3, 3), padding="same")(H)
H = MaxPooling2D((2, 2))(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(32, (3, 3), padding="same")(H)
H = MaxPooling2D((2, 2))(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(64, (3, 3), padding="same", strides=(2, 2))(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(0.001))
discriminator.summary()

def plot_loss(losses):
    with PdfPages('gan_summary.pdf') as pp:
        plt.plot(losses["d"])
        plt.plot(losses["g"])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.legend(['disc', 'gen'], loc='upper left')
        pp.savefig()
        plt.close()

weights_file_name = './d_pre_weights.hdf5'
if os.path.isfile(weights_file_name) == False:
    ntrain = 30000
    # we geneate 10000 indexes of the size of mnist training set.
    train_input_idx = random.sample(range(0, X_train.shape[0]), ntrain)
    trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
    # instead of taking a range we take 100000 random indexes
    XT_input = x_train_input[train_input_idx, :, :, :]
    XT = X_train[trainidx, :, :, :]
    # transform the noise[100] into an image
    generated_images = generator.predict(XT_input)
    # create a new set for discriminator containing generated images and real images
    X = np.concatenate((XT, generated_images))
    n = XT.shape[0]
    # 1 if real, 0 if fake image
    y = np.zeros([2 * n, 2])
    y[:n, 1] = 1
    y[n:, 0] = 1

    y_hat_before = discriminator.predict(X)
    # train discriminator for 1 epoch
    make_disciminator_trainable(discriminator, True)
    checkpoint = ModelCheckpoint(filepath=weights_file_name, verbose=1)
    discriminator.fit(X, y, epochs=1, shuffle=True, batch_size=200, callbacks=[checkpoint])

    # predict fake or real for the new training set created above
    y_hat = discriminator.predict(X)

    # Measure accuracy of pre-trained discriminator network
    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff == 0).sum()
    acc = n_rig * 100.0 / n_tot

    if logs_enabled:
        loss_logs.write("Accuracy: %0.02f pct (%d of %d) right \n" % (acc, n_rig, n_tot))

else:
    print 'Loading saved weights...'
    discriminator.load_weights(weights_file_name)

# Build stacked GAN model
gan_input = Input(shape=(64, 64, 3))
H = generator(gan_input)
gan_V = discriminator(H)  # set of layers?
GAN = Model(gan_input, gan_V)
opt = Adam(0.00001)
GAN.compile(loss='categorical_crossentropy', optimizer=opt)
GAN.summary()

# set up loss storage vector
losses = {"d": [], "g": []}

# Set up our main training loop
def train_for_n(nb_iterations=100, BATCH_SIZE=150):

    for e in range(nb_iterations):
        start = time.time()
        print 'iteration %d' % e
        print 'training discriminator...'
        disc_batches = 5
        disc_batch_size = disc_batches * BATCH_SIZE
        # Make generative images. X_train.shape[0] = 82611, we take out of this a random batch of 32
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=disc_batch_size), :, :, :]
        image_input_batch = x_train_input[np.random.randint(0, X_train.shape[0], size=disc_batch_size), :, :, :]
        generated_images = generator.predict(image_input_batch)

        # Train discriminator on generated images, if real then [0, 1], if fake [1, 0]
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * disc_batch_size, 2])
        y[0:disc_batch_size, 1] = 1 # index 1 corresponds to real images, from 0 to 31
        y[disc_batch_size:, 0] = 1 # index 0 corresponds to fake images, from 32 to 63

        make_disciminator_trainable(discriminator, True)
        # y1_hat_before = discriminator.predict(X)

        shuffled_indexes = np.arange(disc_batch_size * 2)
        np.random.shuffle(shuffled_indexes)
        for b in range(disc_batches):
            subX1 = BATCH_SIZE * 2 * b
            subX2 = BATCH_SIZE * 2 * (b + 1)
            d_loss = discriminator.train_on_batch(X[shuffled_indexes[subX1:subX2]], y[shuffled_indexes[subX1:subX2]])

        # d_history = discriminator.fit(X, y, epochs=1, batch_size=200, shuffle=True)
        # d_loss = d_history.history['loss'][0]
        # y1_hat = discriminator.predict(X)

        print 'd_loss : %0.02f' % d_loss
        losses["d"].append(float(d_loss))

        if logs_enabled:
            discriminator.save_weights('./logs/dis_weights_{i}.hdf5'.format(i=e))

        print 'training generator...'
        # train Generator-Discriminator stack on input noise to non-generated output class
        image_input_batch_tr = x_train_input[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1 # we set it as if it was real and not fake

        make_disciminator_trainable(discriminator, False)
        # y2_hat_before = GAN.predict(image_input_batch_tr)
        # y2_hat_loss = keras.losses.categorical_crossentropy(y2, y2_hat_before).eval()
        g_loss = GAN.train_on_batch(image_input_batch_tr, y2)
        # y2_hat = GAN.predict(image_input_batch_tr)
        print 'g_loss : %0.02f' % g_loss
        losses["g"].append(float(g_loss))

        end = time.time()
        print 'duration : %0.02f' % (end - start)
        print '============================='
        if logs_enabled:
            generator.save_weights('./logs/gen_weights_{i}.hdf5'.format(i=e))
            # discriminator.save_weights('./logs/dis_weights_{i}_postgen.hdf5'.format(i=e))

        if logs_enabled:
            loss_logs.write('epoch: %d - d_loss: %0.02f - g_loss: %0.02f \n' % (e, d_loss, g_loss))



# Train for 6000 epochs at original learning rates
train_for_n(nb_iterations=1000, BATCH_SIZE=150)
plot_loss(losses)

n_ex = 100
image_input_batch_test = x_test_input[0:n_ex, :, :, :]
generated_images = generator.predict(image_input_batch_test)
reshaped_decoded_imgs = generated_images.reshape(n_ex, 64, 64, 3) * 255.
reshaped_decoded_imgs = reshaped_decoded_imgs.astype('uint8')
rs.write_images_to_pkl(reshaped_decoded_imgs, 'gan_decoded.pkl')

if logs_enabled:
    loss_logs.close()
