# if you want to run this on GPU
# THEANO_FLAGS="device=gpu,floatX=float32" ENV\Scripts\python.exe autoencoder.py
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Dropout, \
    GaussianNoise, LeakyReLU, Deconv2D, Flatten, Reshape, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import readsample as rs
import metrics
import os.path
import shutil
import keras

weights_file_name = './tmp/g_pre_weights.hdf5'


def model():
    dropout_rate = 0.1
    leaking_factor = 0.1
    input_img = Input(shape=(64, 64, 3))
    x = input_img
    # Phase1: trained during  generator pre training but not during gan training
    x = Conv2D(8, 3, padding='same', strides=2)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(16, 3, padding='same', strides=2)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(32, 3, padding='same', strides=2)(x)
    x = LeakyReLU(0)(x)
    x = Conv2D(64, 3, padding='same', strides=2)(x)
    x = LeakyReLU(0)(x)

    input_noise = Input(shape=[100])
    noise = Dense(4 * 4 * 64)(input_noise)
    noise = LeakyReLU(leaking_factor)(noise)
    noise = Reshape([4, 4, 64])(noise)
    x = Concatenate()([x, noise])

    # Phase 2
    x = Deconv2D(64, 3, padding='same')(x)
    x = LeakyReLU(leaking_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Deconv2D(32, 3, padding='same', strides=2)(x)
    x = LeakyReLU(leaking_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Deconv2D(16, 3, padding='same', strides=2)(x)
    x = LeakyReLU(leaking_factor)(x)
    x = Dropout(dropout_rate)(x)
    x = Deconv2D(8, 3, padding='same', strides=2)(x)
    x = LeakyReLU(leaking_factor)(x)
    x = Dropout(dropout_rate)(x)

    decoded = Deconv2D(3, 3, activation='sigmoid', strides=2, padding='same')(x)

    # this model maps an input to its reconstruction
    autoencoder = Model([input_img, input_noise], decoded)

    return autoencoder

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


def train(autoencoder, train_input, x_train_target, test_input, x_test_target):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=weights_file_name, verbose=1, save_best_only=True)

    history = autoencoder.fit(train_input, x_train_target,
                              epochs=100,
                              batch_size=250,
                              shuffle=True,
                              validation_data=(test_input, x_test_target),
                              callbacks=[early_stopping, checkpoint])

    return history


def predict(autoencoder, x_test_input, x_test_target):
    # writing predictions to disk
    decoded_imgs = autoencoder.predict(x_test_input)
    reshaped_decoded_imgs = decoded_imgs.reshape(len(x_test_target), 64, 64, 3) * 255.
    reshaped_decoded_imgs = reshaped_decoded_imgs.astype('uint8')
    rs.write_images_to_pkl(reshaped_decoded_imgs, './tmp/gen_pre_train_decoded.pkl')


def train_and_predict():
    x_train_input, x_train_target, x_test_input, x_test_target = load_data()
    autoencoder = model()
    noise_train_input = np.random.uniform(0, 1, size=[x_train_input.shape[0], 100])
    noise_test_input = np.random.uniform(0, 1, size=[x_test_input.shape[0], 100])
    if os.path.isfile(weights_file_name):
        autoencoder.load_weights(weights_file_name)
    else:
        tmpDir = './tmp'
        if os.path.exists(tmpDir):
            shutil.rmtree(tmpDir)
        os.makedirs(tmpDir)
        autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        print 'starting training...'
        autoencoder.summary()

        # JUST A TEST TO SEE IF CALCULATED LOSS ON PREDICTIONS IS CLOSE TO OUPUT LOSS
        # for i in range(1, 100):
        #     y_hat = autoencoder.predict([x_train_input[0:150], noise_train_input[0:150]])  # TO REMOVE
        #     losses = keras.losses.binary_crossentropy(x_train_target[0:150], y_hat).eval()  # TO REMOVE
        #     av_loss = np.mean(losses)  # TO REMOVE
        #     d_loss = autoencoder.train_on_batch([x_train_input[0:150], noise_train_input[0:150]],
        #                                         x_train_target[0:150])  # TO REMOVE
        #     print av_loss
        #     print d_loss
        #     print '==========='


        history = train(autoencoder, [x_train_input, noise_train_input], x_train_target,
                        [x_test_input, noise_test_input], x_test_target)
        print(history.history.keys())
        metrics.plotSaveLossAndAccuracy('./tmp/gen_pre_train_summary.pdf', history)

    print 'starting predicting...'
    predict(autoencoder, [x_test_input, noise_test_input], x_test_target)


def make_generator_phase1_trainable(net, val):
    for i in range(1, 9):
        i_name = 'phase1_%d' % i
        l = net.get_layer(i_name)
        l.trainable = val


if __name__ == "__main__":
    train_and_predict()
