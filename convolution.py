# if you want to run this on GPU
#THEANO_FLAGS="device=gpu,floatX=float32" ENV\Scripts\python.exe autoencoder.py
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import readsample as rs
import metrics


input_img = Input(shape=(64, 64, 3))

x = Conv2D(16, (3, 3), dilation_rate=(4, 4), padding='same')(input_img)
x = Activation('relu')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(32, (3, 3),  dilation_rate=(3, 3), padding='same')(x)
x = Activation('relu')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3),  dilation_rate=(2, 2), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation('relu')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), padding='same')(x)
x = Activation('relu')(x)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

autoencoder.summary()

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

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = autoencoder.fit(x_train_input, x_train_target,
                epochs=100,
                batch_size=300,
                shuffle=True,
                validation_data=(x_test_input, x_test_target),
                callbacks=[early_stopping])

#writing predictions to disk
decoded_imgs = autoencoder.predict(x_test_input)
reshaped_decoded_imgs = decoded_imgs.reshape(len(x_test_target), 64, 64, 3) * 255.
reshaped_decoded_imgs = reshaped_decoded_imgs.astype('uint8')
rs.write_images_to_pkl(reshaped_decoded_imgs, 'convolution_decoded.pkl')

# list all data in history
print(history.history.keys())
metrics.plotSaveLossAndAccuracy('convolution_summary.pdf', history)
