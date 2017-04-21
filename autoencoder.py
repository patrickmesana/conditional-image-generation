# if you want to run this on GPU
#THEANO_FLAGS="device=gpu,floatX=float32" ENV\Scripts\python.exe autoencoder.py
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import readsample as rs
import metrics

img_resolution = 64*64*3 # 12,288
middle_resolution = 32*32*3 # 3,072
# this is the size of our encoded representations
encoding_dim = img_resolution  # compression factor of 2
# this is our input placeholder (we want to keep a reference to this, although it's part of the Model)
input_img = Input(shape=(img_resolution,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
encoded = Dense(encoding_dim/2, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(middle_resolution, activation='relu')(encoded)
decoded = Dense(middle_resolution, activation='sigmoid')(decoded)

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
x_train_input = x_train_input.reshape((len(x_train_input), np.prod(x_train_input.shape[1:])))
x_train_target = x_train_target.reshape((len(x_train_target), np.prod(x_train_target.shape[1:])))
x_test_input = x_test_input.reshape((len(x_test_input), np.prod(x_test_input.shape[1:])))
x_test_target = x_test_target.reshape((len(x_test_target), np.prod(x_test_target.shape[1:])))

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
