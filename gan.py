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
import gan_generator

tmpDir = './tmp'
if os.path.exists(tmpDir):
    shutil.rmtree(tmpDir)
os.makedirs(tmpDir)

if os.path.exists('./tmp/gan_decoded.pkl'):
    os.remove('./tmp/gan_decoded.pkl')
if os.path.exists('./tmp/gan_summary.pdf'):
    os.remove('./tmp/gan_summary.pdf')
logs_enabled = True
if logs_enabled:
    # removes logs if necessary and create it
    dir = './tmp/logs'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    loss_logs = open('./tmp/logs/loss_logs.csv', 'w')


def show_denormalized(image, i=0):
    Image.fromarray((image * 255).astype('uint8')).save('./tmp/PredictedImage{i}.jpg'.format(i=i))


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
generator = gan_generator.model()
# generator.load_weights('./g_pre_weights.hdf5')
genOpt = Adam(0.00001)
# gan_generator.make_generator_phase1_trainable(generator, False)
generator.compile(optimizer=genOpt, loss='binary_crossentropy')
generator.summary()

# Build Discriminative model ...
dropout_rate = 0.2
leaky_factor = 0.1
d_input = Input(shape=shp)
H = Convolution2D(8, 3, strides=2, padding='same')(d_input)
H = LeakyReLU(leaky_factor)(H)
# H = Dropout(dropout_rate)(H)
H = Convolution2D(16, 3, strides=2, padding='same')(H)
H = LeakyReLU(leaky_factor)(H)
# H = Dropout(dropout_rate)(H)
H = Convolution2D(32, 3, strides=2, padding='same')(H)
H = LeakyReLU(leaky_factor)(H)
# H = Dropout(dropout_rate)(H)
H = Convolution2D(64, 3, strides=2, padding='same')(H)
H = LeakyReLU(leaky_factor)(H)
# H = Dropout(dropout_rate)(H)
H = Flatten()(H)
# H = Dropout(dropout_rate)(H)
H = Dense(256)(H)
H = LeakyReLU(leaky_factor)(H)
# H = Dropout(dropout_rate)(H)
d_V = Dense(2, activation='softmax')(H)
discriminator = Model(d_input, d_V)
discriminator.compile(loss='mean_squared_error', optimizer=Adam(0.001))
discriminator.summary()

def clear_plot():
    plt.clf()
    plt.cla()
    plt.close()

def plot_loss(losses):
    plt.plot(losses["d"])
    plt.plot(losses["g"])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend(['disc', 'gen'], loc='upper left')
    plt.show(block=False)


def save_plot_to_pdf():
    with PdfPages('./tmp/gan_summary.pdf') as pp:
        pp.savefig()
        plt.close()

# discriminator pre-training
# weights_file_name = './d_pre_weights.hdf5' #936 from previous
# if os.path.isfile(weights_file_name) == False:
#     ntrain = 40000
#     # we geneate 10000 indexes of the size of mnist training set.
#     train_input_idx = random.sample(range(0, X_train.shape[0]), ntrain)
#     trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
#     # instead of taking a range we take 100000 random indexes
#     XT_input = x_train_input[train_input_idx, :, :, :]
#     XT = X_train[trainidx, :, :, :]
#     n = XT.shape[0]
#     # transform the noise[100] into an image
#     noise_XT = np.random.uniform(0, 1, size=[ntrain, 100])
#     generated_images = generator.predict([XT_input, noise_XT])
#     # create a new set for discriminator containing generated images and real images
#     X = np.concatenate((XT, generated_images))
#
#     # 1 if real, 0 if fake image
#     y = np.zeros([2 * n, 2])
#     y[:n, 1] = 1
#     y[n:, 0] = 1
#
#     # y_hat_before = discriminator.predict(X)
#     # train discriminator for 1 epoch
#     make_disciminator_trainable(discriminator, True)
#     checkpoint = ModelCheckpoint(filepath=weights_file_name, verbose=1)
#     discriminator.fit(X, y, epochs=1, shuffle=True, batch_size=200, callbacks=[checkpoint])
#
#     # predict fake or real for the new training set created above
#     y_hat = discriminator.predict(X)
#
#     # Measure accuracy of pre-trained discriminator network
#     y_hat_idx = np.argmax(y_hat, axis=1)
#     y_idx = np.argmax(y, axis=1)
#     diff = y_idx - y_hat_idx
#     n_tot = y.shape[0]
#     n_rig = (diff == 0).sum()
#     acc = n_rig * 100.0 / n_tot
#
#     print "Accuracy: %0.02f pct (%d of %d) right \n" % (acc, n_rig, n_tot)
#
# else:
#     print 'Loading saved weights...'
#     discriminator.load_weights(weights_file_name)

# Build stacked GAN model
gan_input = Input(shape=(64, 64, 3))
noise_shape = Input(shape=[100])
H = generator([gan_input, noise_shape])
gan_V = discriminator(H)  # set of layers?
GAN = Model([gan_input, noise_shape], gan_V)
GAN.compile(loss='mean_squared_error', optimizer=genOpt)
GAN.summary()

# set up loss storage vector
losses = {"d": [], "g": []}

# Set up our main training loop
def train_for_n(nb_iterations=100, BATCH_SIZE=150):

    for e in range(nb_iterations):
        start = time.time()
        print 'iteration %d' % e
        print 'training discriminator...'

        if e > 0:
            disc_batches = 2
        else:
            disc_batches = 1000

        disc_batch_size = disc_batches * BATCH_SIZE
        # Make generative images. X_train.shape[0] = 82611, we take out of this a random batch of 32
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=disc_batch_size), :, :, :]
        image_input_batch = x_train_input[np.random.randint(0, X_train.shape[0], size=disc_batch_size), :, :, :]

        noise_X = np.random.uniform(0, 1, size=[disc_batch_size, 100])

        generated_images = generator.predict([image_input_batch, noise_X])

        Image.fromarray((image_batch[0] * 255).astype('uint8')).show()
        Image.fromarray((generated_images[0] * 255).astype('uint8')).show()

        if e % 10 == 0:
            noise5 = np.random.uniform(0, 1, size=[5, 100])
            show_denormalized(generator.predict([x_test_input[0:5], noise5])[3], e)

        # Train discriminator on generated images, if real then [0, 1], if fake [1, 0]
        X_1 = image_batch
        X_2 = generated_images
        y_1 = np.zeros([disc_batch_size, 2])
        y_2 = np.zeros([disc_batch_size, 2])
        y_1[:, 1] = 1
        y_2[:, 0] = 1

        make_disciminator_trainable(discriminator, True)

        # y1_hat_before = discriminator.predict(X)

        shuffled_indexes = np.arange(disc_batch_size)
        np.random.shuffle(shuffled_indexes)
        d_losses = []
        for b in range(disc_batches):
            subX1 = BATCH_SIZE * b
            subX2 = BATCH_SIZE * (b + 1)

            if b % 2 == 0:
                sub_batch1 = X_1[shuffled_indexes[subX1:subX2]]
                sub_target1 = y_1[shuffled_indexes[subX1:subX2]]
                sub_noise = np.random.uniform(0, 1, size=[BATCH_SIZE, 100]) # TO REMOVE
                sub_predict1 = GAN.predict([sub_batch1, sub_noise])  # TO REMOVE
                sub_y_hat_loss1 = keras.losses.mean_squared_error(sub_target1, sub_predict1).eval()  # TO REMOVE
                av_loss1 = sum(sub_y_hat_loss1) / len(sub_y_hat_loss1)
                d_loss1 = discriminator.train_on_batch(sub_batch1, sub_target1)
                d_loss = d_loss1
            else:
                sub_batch2 = X_2[shuffled_indexes[subX1:subX2]]
                sub_target2 = y_2[shuffled_indexes[subX1:subX2]]
                sub_noise = np.random.uniform(0, 1, size=[BATCH_SIZE, 100]) # TO REMOVE
                sub_predict2 = GAN.predict([sub_batch2, sub_noise])  # TO REMOVE
                sub_y_hat_loss2 = keras.losses.categorical_crossentropy(sub_target2, sub_predict2).eval()  # TO REMOVE
                d_loss2 = discriminator.train_on_batch(sub_batch2, sub_target2)
                d_loss = d_loss2
            print 'it:%d d_loss:%f' % (b, d_loss)
            d_losses.append(d_loss)

        test_noise11 = np.random.uniform(0, 1, size=[disc_batch_size, 100]) #TO REMOVE
        y11_hat = GAN.predict([X_1, test_noise11])#TO REMOVE
        y11_hat_loss = keras.losses.categorical_crossentropy(y_1, y11_hat).eval()  # TO REMOVE
        test_noise12 = np.random.uniform(0, 1, size=[disc_batch_size, 100]) #TO REMOVE
        y12_hat = GAN.predict([X_2, test_noise12])#TO REMOVE
        y12_hat_loss = keras.losses.categorical_crossentropy(y_2, y12_hat).eval()  # TO REMOVE

        print 'd_losses : ' + str([float(l) for l in d_losses])
        losses["d"].append(float(d_loss))

        if logs_enabled:
            discriminator.save_weights('./tmp/logs/d_weights_{i}.hdf5'.format(i=e))

        print 'training generator...'
        make_disciminator_trainable(discriminator, False)
        gen_batches = 1
        gen_batch_size = gen_batches * BATCH_SIZE
        # train Generator-Discriminator stack on input noise to non-generated output class
        image_input_batch_tr = x_train_input[np.random.randint(0, X_train.shape[0], size=gen_batch_size), :, :, :]

        y2 = np.zeros([gen_batch_size, 2])
        y2[:, 1] = 1 # we set it as if it was real and not fake

        g_shuffled_indexes = np.arange(gen_batch_size)
        np.random.shuffle(g_shuffled_indexes)

        test_noise2 = np.random.uniform(0, 1, size=[gen_batch_size, 100]) #TO REMOVE
        y2_hat_before = GAN.predict([image_input_batch_tr, test_noise2])#TO REMOVE
        y2_hat_loss = keras.losses.categorical_crossentropy(y2, y2_hat_before).eval()  # TO REMOVE

        g_losses = []
        for b in range(gen_batches):
            g_subX1 = BATCH_SIZE * b
            g_subX2 = BATCH_SIZE * (b + 1)
            g_noise_X = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
            g_loss = GAN.train_on_batch([image_input_batch_tr[g_shuffled_indexes[g_subX1:g_subX2]], g_noise_X], y2[g_shuffled_indexes[g_subX1:g_subX2]])
            g_losses.append(g_loss)


        y2_hat = GAN.predict([image_input_batch_tr, test_noise2])#TO REMOVE

        print 'g_losses : ' + str([float(l) for l in g_losses])
        losses["g"].append(float(g_loss))
        if e > 0:
            clear_plot()
        plot_loss(losses)
        end = time.time()
        print 'duration : %0.02f' % (end - start)
        print '============================='
        if logs_enabled:
            generator.save_weights('./tmp/logs/g_weights_{i}.hdf5'.format(i=e))
            # discriminator.save_weights('./logs/dis_weights_{i}_postgen.hdf5'.format(i=e))

        if logs_enabled:
            loss_logs.write('%d%0.02f,%0.02f\n' % (e, d_loss, g_loss))



# Train for 6000 epochs at original learning rates
train_for_n(nb_iterations=1000, BATCH_SIZE=32)
save_plot_to_pdf()

n_ex = 100
image_input_batch_test = x_test_input[0:n_ex, :, :, :]
noise_test_input = np.random.uniform(0, 1, size=[n_ex, 100])
generated_images = generator.predict([image_input_batch_test, noise_test_input])
reshaped_decoded_imgs = generated_images.reshape(n_ex, 64, 64, 3) * 255.
reshaped_decoded_imgs = reshaped_decoded_imgs.astype('uint8')
rs.write_images_to_pkl(reshaped_decoded_imgs, './tmp/gan_decoded.pkl')

if logs_enabled:
    loss_logs.close()