# if you want to run this on GPU
#THEANO_FLAGS="device=gpu,floatX=float32" ENV\Scripts\python.exe autoencoder.py
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import numpy as np
import readsample as rs
import metrics


input_img = Input(shape=(64, 64, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

x_train_input = rs.read_images_from_pkl('training_input.pkl')
x_train_target = rs.read_images_from_pkl('training_target.pkl')
x_test_input_unchanged = rs.read_images_from_pkl('validation_input.pkl')
x_test_target = rs.read_images_from_pkl('validation_target.pkl')
x_train_input = x_train_input.astype('float32') / 255.
x_train_target = x_train_target.astype('float32') / 255.
x_test_input = x_test_input_unchanged.astype('float32') / 255.
x_test_target = x_test_target.astype('float32') / 255.
x_train_input = x_train_input.reshape((len(x_train_input), 64, 64, 3))
x_train_target = x_train_target.reshape((len(x_train_target), 64, 64, 3))
x_test_input = x_test_input.reshape((len(x_test_input), 64, 64, 3))
x_test_target = x_test_target.reshape((len(x_test_target), 64, 64, 3))

history = autoencoder.fit(x_train_input, x_train_target,
                epochs=60,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_input, x_test_target))

#writing predictions to disk
decoded_imgs = autoencoder.predict(x_test_input)
reshaped_decoded_imgs = decoded_imgs.reshape(len(x_test_target), 32, 32, 3) * 255.
reshaped_decoded_imgs = reshaped_decoded_imgs.astype('uint8')
rs.write_images_to_pkl(reshaped_decoded_imgs, 'autoencoder_decoded.pkl')

# list all data in history
print(history.history.keys())
metrics.plotSaveLossAndAccuracy('autoencoder_summary.pdf', history)
