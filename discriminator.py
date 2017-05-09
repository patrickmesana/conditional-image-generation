from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2D
from keras.models import Model
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os, random, shutil
import numpy as np
import readsample as rs
import generator
from keras import backend as k
import theano.tensor as T

tmp_weights_file_name = './tmp/d_pre_weights.hdf5'
weights_file_name = './d_pre_weights.hdf5'
gen_weights_file_name = './g_pre_weights.hdf5'


def train(gen, disc, training_set_cropped, training_set_full, unsafe_train_n, batch_size=100, epochs=1, accuracy_enable=False):

    train_n = int(unsafe_train_n / batch_size) * batch_size
    training_set_size = training_set_full.shape[0]

    # Creating a set of random index in the range of 0:ntrain
    shuffled_indexes = np.arange(training_set_size)
    np.random.shuffle(shuffled_indexes)
    training_indexes = shuffled_indexes[0:train_n]

    # Selecting the images (full and cropped) for training
    training_cropped_selected = training_set_cropped[training_indexes, :, :, :]
    training_full_selected = training_set_full[training_indexes, :, :, :]

    # transform the noise[100] into an image
    noise = np.random.uniform(0, 1, size=[train_n, 100])
    generated_images = gen.predict([training_cropped_selected, noise])
    x, y = build_real_and_fake_sets(training_full_selected, generated_images)

    # Fitting
    d_loss = fit(disc, x, y, batch_size, epochs=epochs)

    if accuracy_enable:
        # Measure accuracy of pre-trained discriminator network
        y_hat = disc.predict(x)
        accuracy(y, y_hat)

        acc, n_rig, n_tot = accuracy(y, y_hat)
        print "Accuracy: %0.02f pct (%d of %d) right \n" % (acc, n_rig, n_tot)
    return d_loss


def create_noise_images(train_n):
    return np.random.uniform(0, 1, size=[train_n, 64 * 64 * 3]).reshape((train_n, 64, 64, 3))


def build_real_and_fake_sets(training_full_selected, generated_images):
    x = np.concatenate((training_full_selected, generated_images))
    train_n = training_full_selected.shape[0]
    # Creating target [0, 1] (real) or [1, 0] (fake)
    y = np.zeros([train_n * 2, 2])
    y[:train_n, 1] = 1
    y[train_n:, 0] = 1
    return x, y


def accuracy(y, y_hat):
    y_hat_idx = np.argmax(y_hat, axis=1)
    y_idx = np.argmax(y, axis=1)
    diff = y_idx - y_hat_idx
    n_tot = y.shape[0]
    n_rig = (diff == 0).sum()
    acc = n_rig * 100.0 / n_tot
    return acc, n_rig, n_tot


def fit(disc, x, y, batch_size, epochs=1):
    checkpoint = ModelCheckpoint(filepath=tmp_weights_file_name, verbose=1)
    d_loss = disc.fit(x, y, epochs=epochs, shuffle='batch', batch_size=batch_size, callbacks=[checkpoint])
    return d_loss


def train_batches(disc_batches, disc, batch_size, X_train, generated_images):
    print 'starting to train discriminator, BATCH_SIZE: %d' % batch_size
    disc_batch_size = disc_batches * batch_size
    image_batch = X_train[np.random.randint(0, X_train.shape[0], size=disc_batch_size), :, :, :]

    # Train discriminator on generated images, if real then [0, 1], if fake [1, 0]
    X_1 = image_batch
    X_2 = generated_images
    y_1 = np.zeros([disc_batch_size, 2])
    y_2 = np.zeros([disc_batch_size, 2])
    y_1[:, 1] = 1
    y_2[:, 0] = 1

    shuffled_indexes = np.arange(disc_batch_size)
    np.random.shuffle(shuffled_indexes)
    d_losses = []
    for b in range(disc_batches):
        sub_x_lo = batch_size * b
        sub_x_up = batch_size * (b + 1)

        if b % 2 == 0:
            sub_batch1 = X_1[shuffled_indexes[sub_x_lo:sub_x_up]]
            sub_target1 = y_1[shuffled_indexes[sub_x_lo:sub_x_up]]
            d_loss1 = disc.train_on_batch(sub_batch1, sub_target1)
            d_loss = d_loss1
            print '(real) it:%d d_loss:%f' % (b, d_loss)
        else:
            sub_batch2 = X_2[shuffled_indexes[sub_x_lo:sub_x_up]]
            sub_target2 = y_2[shuffled_indexes[sub_x_lo:sub_x_up]]
            d_loss2 = disc.train_on_batch(sub_batch2, sub_target2)
            d_loss = d_loss2
            print '(fake) it:%d d_loss:%f' % (b, d_loss)

        d_losses.append(d_loss)
    return d_losses


def train_batches_with_generated_images(gen, x_train_input, X_train, disc_batch_size):
    """
    NOT IMPLEMENTED, JUST A DRAFT
    """
    # output of gen is input of disc
    image_input_batch = x_train_input[np.random.randint(0, X_train.shape[0], size=disc_batch_size), :, :, :]
    noise_X = np.random.uniform(0, 1, size=[disc_batch_size, 100])
    generated_images = gen.predict([image_input_batch, noise_X])
    train_batches(0)


def model():
    dropout_rate = 0.2
    leaky_factor = 0.1
    d_input = Input(shape=(64, 64, 3))
    x = Convolution2D(8, 3, strides=2, padding='same')(d_input)
    x = LeakyReLU(leaky_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Convolution2D(16, 3, strides=2, padding='same')(x)
    x = LeakyReLU(leaky_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Convolution2D(32, 3, strides=2, padding='same')(x)
    x = LeakyReLU(leaky_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Convolution2D(64, 3, strides=2, padding='same')(x)
    x = LeakyReLU(leaky_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256)(x)
    x = LeakyReLU(leaky_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(2, activation='softmax')(x)
    disc = Model(d_input, x)
    return disc


def delete_and_create_tmp():
    tmp_dir = './tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


def load_data():
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
    return x_train_input, x_train_target, x_test_input, x_test_target


def main():
    delete_and_create_tmp()
    disc = model()
    disc.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    disc.summary()
    gen = generator.model()

    # gen.load_weights(gen_weights_file_name)
    # disc.load_weights(weights_file_name)

    gen.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    x_train_input, x_train_target, x_test_input, x_test_target = load_data()

    train(gen, disc, x_train_input, x_train_target, 82000, accuracy_enable=True, epochs=1)

    # print 'Data loaded'
    # if not os.path.isfile(weights_file_name):
    #     train(gen, disc, x_train_input, x_train_target, 50000, accuracy_enable=True, epochs=10)
    # else:
    #     print 'Loading saved weights...'
    #     disc.load_weights(weights_file_name)
    #     n = x_test_input.shape[0]
    #     noise = np.random.uniform(0, 1, size=[n, 100])
    #     generated_images = gen.predict([x_test_input, noise])
    #     x, y = build_real_and_fake_sets(x_test_input, generated_images)
    #     y_hat = disc.predict(x)
    #     acc, _, _ = accuracy(y, y_hat)
    #
    #     print acc


def calculate_gradients(disc, x, y):
    # outputTensor = disc.output
    # listOfVariableTensors = disc.trainable_weights
    # gradients = k.gradients(outputTensor[0][0], listOfVariableTensors)


    weights = disc.trainable_weights  # weight tensors
    # weights = [weight for weight in weights if model.get_layer(
    #     weight.name[:-2]).trainable]  # filter down weights tensors to only ones which are trainable
    gradients = disc.optimizer.get_gradients(disc.total_loss, weights)  # gradient tensors

    print weights
    # ==> [dense_1_W, dense_1_b]
    import keras.backend as K

    input_tensors = [disc.inputs[0],  # input data
                     disc.sample_weights[0],  # how much to weight each sample by
                     disc.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    get_gradients = K.function(inputs=input_tensors, outputs=gradients)

    from keras.utils.np_utils import to_categorical

    inputs = [x[0:5],  # X
              [1],  # sample weights
              y[0:5],  # y
              0  # learning phase in TEST mode
              ]

    print weights
    print zip(weights, get_gradients(inputs))
    # ==> [(dense_1_W, array([[-0.42342907],
    # [-0.84685814]], dtype = float32)),
    # (dense_1_b, array([-0.42342907], dtype=float32))]


if __name__ == "__main__":
    main()
