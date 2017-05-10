#
# Keras GAN Implementation
# Forked from: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/
#
import os, random, shutil
import numpy as np
from keras.layers import Input, Lambda
from keras.optimizers import *
import matplotlib.pyplot as plt
import cPickle, random, sys, keras
from keras.models import Model
import readsample as rs
import display
from matplotlib.backends.backend_pdf import PdfPages
import PIL.Image as Image
from keras import losses
import time
import generator
import discriminator
from keras import backend as k
import theano.tensor as T

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


def show_denormalized(image, i=0, saving=True):
    img = Image.fromarray((image * 255).astype('uint8'))
    if saving:
        img.save('./tmp/PredictedImage{i}.jpg'.format(i=i))
    else:
        img.show()


x_train_input = rs.read_images_from_pkl('training_input.pkl')
x_train_target = rs.read_images_from_pkl('training_target_full.pkl')
x_test_input_unchanged = rs.read_images_from_pkl('validation_input.pkl')
x_test_target = rs.read_images_from_pkl('validation_target_full.pkl')
x_train_input = x_train_input.astype('float32') / 255.
x_train_target = x_train_target.astype('float32') / 255.
x_test_input = x_test_input_unchanged.astype('float32') / 255.
x_test_target = x_test_target.astype('float32') / 255.
x_train_input = x_train_input.reshape((len(x_train_input), 64, 64, 3))
x_train_target = x_train_target.reshape((len(x_train_target), 64, 64, 3))
x_test_input = x_test_input.reshape((len(x_test_input), 64, 64, 3))
x_test_target = x_test_target.reshape((len(x_test_target), 64, 64, 3))


X_train = x_train_target
disc_learning_rate = 0.00001
gen_learning_rate = disc_learning_rate


# the weights are not updated, need to compile again to take into account
# Freeze weights in the discriminator for stacked training
def make_disciminator_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
    net.compile(loss='categorical_crossentropy', optimizer=Adam(disc_learning_rate))


gen = generator.model()
genOpt = Adam(gen_learning_rate)

gen_weights_file_name = './g_pre_weights.hdf5'
if os.path.isfile(gen_weights_file_name):
    print 'Loading saved gen weights...'
    pretrained_gen = generator.model()
    pretrained_gen.load_weights(gen_weights_file_name)
    pretrained_gen.compile(optimizer=genOpt, loss='binary_crossentropy')
    pre_trained_phase1_weights = generator.get_generator_phase1_weights(pretrained_gen)
    generator.set_generator_phase1_weights(gen, pre_trained_phase1_weights)

# generator.make_generator_phase1_trainable(gen, False)

gen.compile(optimizer=genOpt, loss='binary_crossentropy')
gen.summary()

# Build Discriminative model ...
disc = discriminator.model()
weights_file_name = './d_pre_weights.hdf5'
if os.path.isfile(weights_file_name):
    print 'Loading saved disc weights...'
    disc.load_weights(weights_file_name)

disc.compile(loss='categorical_crossentropy', optimizer=Adam(disc_learning_rate))
disc.summary()


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


# Build stacked GAN model
gan_input = Input(shape=(64, 64, 3))
gan_output = Input(shape=(2,))
gen_output = Input(shape=(64, 64, 3))
noise_shape = Input(shape=[100])
H = gen([gan_input, noise_shape])
gan_V = disc(H)


def customized_loss(args):
    gan_arg, gan_target_arg, gen_arg, gen_target = args
    custom_disc_loss = k.categorical_crossentropy(gan_arg, gan_target_arg)
    custom_gen_loss = k.binary_crossentropy(gen_arg, gen_target)
    return custom_gen_loss


loss_out = Lambda(customized_loss, output_shape=(1,), name='joint_loss')([gan_V, gan_output, H, gen_output])
GAN = Model([gan_input, noise_shape, gan_output, gen_output], loss_out)
GAN.compile(loss={'joint_loss': lambda y_true, y_pred: y_pred}, optimizer=genOpt)

# GAN = Model([gan_input, noise_shape], gan_V)
# GAN.compile(loss='categorical_crossentropy', optimizer=genOpt)
GAN.summary()

# set up loss storage vector
losses = {"d": [], "g": []}


def train(gen_model, disc_model, training_set_cropped, training_set_full, nb_iterations=4000, batch_size=150,
          disc_train_batches=1, gen_train_batches=1):
    # number of examples in data to select
    disc_unsafe_train_n = disc_train_batches * batch_size
    gen_unsafe_train_n = gen_train_batches * batch_size

    for e in range(nb_iterations):
        start = time.time()

        # saving predicted images to show progress
        if e % 10 == 0:
            noise5 = np.random.uniform(0, 1, size=[5, 100])
            show_denormalized(gen.predict([x_test_input[0:5], noise5])[3], e)

        # ==== training discriminator =====
        print 'training discriminator'
        make_disciminator_trainable(disc, True)
        # GAN.get_layer('joint_loss').trainable = True
        d_history = discriminator.train(gen_model, disc_model, training_set_cropped, training_set_full,
                                        disc_unsafe_train_n,
                                        batch_size)
        d_loss = d_history.history['loss'][0]
        losses["d"].append(float(d_loss))
        if logs_enabled:
            disc.save_weights('./tmp/logs/d_weights_{i}.hdf5'.format(i=e))

        # ==== training generator =====
        print 'training generator'
        make_disciminator_trainable(disc, False)
        # GAN.get_layer('joint_loss').trainable = False
        g_history = fit(GAN, batch_size, gen_unsafe_train_n, training_set_cropped, training_set_full)

        g_loss = g_history.history['loss'][0]
        losses["g"].append(float(g_loss))
        if logs_enabled:
            gen.save_weights('./tmp/logs/g_weights_{i}.hdf5'.format(i=e))

        end = time.time()
        print 'duration : %0.02f' % (end - start)
        print '============================='

        if logs_enabled:
            loss_logs.write('%d%0.02f,%0.02f\n' % (e, d_loss, g_loss))

        if e > 0:
            clear_plot()
        plot_loss(losses)


def fit(gan_model, batch_size, gen_unsafe_train_n, training_set_cropped, training_set_full):
    # Creating a set of random index in the range of 0:ntrain
    training_set_size = training_set_full.shape[0]
    shuffled_indexes = np.arange(training_set_size)
    np.random.shuffle(shuffled_indexes)
    training_indexes = shuffled_indexes[0:gen_unsafe_train_n]
    # Selecting the images (full and cropped) for training
    training_cropped_selected = training_set_cropped[training_indexes, :, :, :]
    training_full_selected = training_set_full[training_indexes, :, :, :]
    # transform the noise[100] into an image
    noise = np.random.uniform(0, 1, size=[gen_unsafe_train_n, 100])

    train_n = training_full_selected.shape[0]
    y = np.zeros([train_n, 2])
    y[:train_n, 1] = 1

    x = [training_cropped_selected, noise, y, training_full_selected]

    g_history = gan_model.fit(x, y, epochs=1, shuffle='batch', batch_size=batch_size)
    return g_history


# Set up our main training loop
def train_for_n_with_batches(nb_iterations=100, BATCH_SIZE=150):
    """
    NOT IMPLEMENTED
    :param nb_iterations:
    :param BATCH_SIZE:
    :return:
    """
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

        generated_images = gen.predict([image_input_batch, noise_X])

        Image.fromarray((image_batch[0] * 255).astype('uint8')).show()
        Image.fromarray((generated_images[0] * 255).astype('uint8')).show()

        if e % 10 == 0:
            noise5 = np.random.uniform(0, 1, size=[5, 100])
            show_denormalized(gen.predict([x_test_input[0:5], noise5])[3], e)

        # Train discriminator on generated images, if real then [0, 1], if fake [1, 0]
        X_1 = image_batch
        X_2 = generated_images
        y_1 = np.zeros([disc_batch_size, 2])
        y_2 = np.zeros([disc_batch_size, 2])
        y_1[:, 1] = 1
        y_2[:, 0] = 1

        make_disciminator_trainable(disc, True)

        # y1_hat_before = disc.predict(X) #TO REMOVE

        shuffled_indexes = np.arange(disc_batch_size)
        np.random.shuffle(shuffled_indexes)
        d_losses = []
        for b in range(disc_batches):
            subX1 = BATCH_SIZE * b
            subX2 = BATCH_SIZE * (b + 1)

            if b % 2 == 0:
                sub_batch1 = X_1[shuffled_indexes[subX1:subX2]]
                sub_target1 = y_1[shuffled_indexes[subX1:subX2]]
                sub_noise = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])  # TO REMOVE
                sub_predict1 = GAN.predict([sub_batch1, sub_noise])  # TO REMOVE
                sub_y_hat_loss1 = keras.losses.mean_squared_error(sub_target1, sub_predict1).eval()  # TO REMOVE
                av_loss1 = sum(sub_y_hat_loss1) / len(sub_y_hat_loss1)
                d_loss1 = disc.train_on_batch(sub_batch1, sub_target1)
                d_loss = d_loss1
            else:
                sub_batch2 = X_2[shuffled_indexes[subX1:subX2]]
                sub_target2 = y_2[shuffled_indexes[subX1:subX2]]
                sub_noise = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])  # TO REMOVE
                sub_predict2 = GAN.predict([sub_batch2, sub_noise])  # TO REMOVE
                sub_y_hat_loss2 = keras.losses.categorical_crossentropy(sub_target2, sub_predict2).eval()  # TO REMOVE
                d_loss2 = disc.train_on_batch(sub_batch2, sub_target2)
                d_loss = d_loss2
            print 'it:%d d_loss:%f' % (b, d_loss)
            d_losses.append(d_loss)

        test_noise11 = np.random.uniform(0, 1, size=[disc_batch_size, 100])  # TO REMOVE
        y11_hat = GAN.predict([X_1, test_noise11])  # TO REMOVE
        y11_hat_loss = keras.losses.categorical_crossentropy(y_1, y11_hat).eval()  # TO REMOVE
        test_noise12 = np.random.uniform(0, 1, size=[disc_batch_size, 100])  # TO REMOVE
        y12_hat = GAN.predict([X_2, test_noise12])  # TO REMOVE
        y12_hat_loss = keras.losses.categorical_crossentropy(y_2, y12_hat).eval()  # TO REMOVE

        print 'd_losses : ' + str([float(l) for l in d_losses])
        losses["d"].append(float(d_loss))

        if logs_enabled:
            disc.save_weights('./tmp/logs/d_weights_{i}.hdf5'.format(i=e))

        print 'training generator...'
        make_disciminator_trainable(disc, False)
        gen_batches = 1
        gen_batch_size = gen_batches * BATCH_SIZE
        # train Generator-Discriminator stack on input noise to non-generated output class
        image_input_batch_tr = x_train_input[np.random.randint(0, X_train.shape[0], size=gen_batch_size), :, :, :]

        y2 = np.zeros([gen_batch_size, 2])
        y2[:, 1] = 1  # we set it as if it was real and not fake

        g_shuffled_indexes = np.arange(gen_batch_size)
        np.random.shuffle(g_shuffled_indexes)

        test_noise2 = np.random.uniform(0, 1, size=[gen_batch_size, 100])  # TO REMOVE
        y2_hat_before = GAN.predict([image_input_batch_tr, test_noise2])  # TO REMOVE
        y2_hat_loss = keras.losses.categorical_crossentropy(y2, y2_hat_before).eval()  # TO REMOVE

        g_losses = []
        for b in range(gen_batches):
            g_subX1 = BATCH_SIZE * b
            g_subX2 = BATCH_SIZE * (b + 1)
            g_noise_X = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
            g_loss = GAN.train_on_batch([image_input_batch_tr[g_shuffled_indexes[g_subX1:g_subX2]], g_noise_X],
                                        y2[g_shuffled_indexes[g_subX1:g_subX2]])
            g_losses.append(g_loss)

        y2_hat = GAN.predict([image_input_batch_tr, test_noise2])  # TO REMOVE

        print 'g_losses : ' + str([float(l) for l in g_losses])
        losses["g"].append(float(g_loss))
        if e > 0:
            clear_plot()
        plot_loss(losses)
        end = time.time()
        print 'duration : %0.02f' % (end - start)
        print '============================='
        if logs_enabled:
            gen.save_weights('./tmp/logs/g_weights_{i}.hdf5'.format(i=e))
            # disc.save_weights('./logs/dis_weights_{i}_postgen.hdf5'.format(i=e))

        if logs_enabled:
            loss_logs.write('%d%0.02f,%0.02f\n' % (e, d_loss, g_loss))


# Train for 6000 epochs at original learning rates
train(gen, disc, x_train_input, x_train_target)
save_plot_to_pdf()
print 'finished training'

n_ex = 100
image_input_batch_test = x_test_input[0:n_ex, :, :, :]
noise_test_input = np.random.uniform(0, 1, size=[n_ex, 100])
generated_images = gen.predict([image_input_batch_test, noise_test_input])
reshaped_decoded_imgs = generated_images.reshape(n_ex, 64, 64, 3) * 255.
reshaped_decoded_imgs = reshaped_decoded_imgs.astype('uint8')
rs.write_images_to_pkl(reshaped_decoded_imgs, './tmp/gan_decoded.pkl')

if logs_enabled:
    loss_logs.close()
