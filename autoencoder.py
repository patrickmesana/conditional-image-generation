# if you want to run this on GPU
#THEANO_FLAGS="device=gpu,floatX=float32" ENV\Scripts\python.exe autoencoder.py
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np
import readsample as rs
import PIL.Image as Image

img_resolution = 64*64*3 # 12,288
# this is the size of our encoded representations
encoding_dim = img_resolution  # compression factor of 2
# this is our input placeholder (we want to keep a reference to this, although it's part of the Model)
input_img = Input(shape=(img_resolution,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
encoded = Dense(img_resolution/2, activation='relu')(encoded)
encoded = Dense(img_resolution/4, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(img_resolution/2, activation='relu')(encoded)
decoded = Dense(encoding_dim, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded input
encoded_input = Input(shape=(img_resolution/4,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-2]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

x_train_input = rs.read_images_from_pkl('training_input.pkl')
x_train_target = rs.read_images_from_pkl('training_target.pkl')
x_test_input = rs.read_images_from_pkl('validation_input.pkl')
x_test_target = rs.read_images_from_pkl('validation_target.pkl')
x_train_input = x_train_input.astype('float32') / 255.
x_train_target = x_train_target.astype('float32') / 255.
x_test_input = x_test_input.astype('float32') / 255.
x_test_target = x_test_target.astype('float32') / 255.
x_train_input = x_train_input.reshape((len(x_train_input), np.prod(x_train_input.shape[1:])))
x_train_target = x_train_target.reshape((len(x_train_target), np.prod(x_train_target.shape[1:])))
x_test_input = x_test_input.reshape((len(x_test_input), np.prod(x_test_input.shape[1:])))
x_test_target = x_test_target.reshape((len(x_test_target), np.prod(x_test_target.shape[1:])))

autoencoder.fit(x_train_input, x_train_target,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_input, x_test_target))


encoded_imgs = encoder.predict(x_train_input)
decoded_imgs = decoder.predict(encoded_imgs)

original_img = x_test_input.reshape(len(x_test_input), 64, 64, 3) * 255.
original_img = original_img.astype('uint8')
decoded_img = decoded_imgs.reshape(len(decoded_imgs), 64, 64, 3) * 255.
decoded_img = decoded_img.astype('uint8')
Image.fromarray(original_img[0]).show()
Image.fromarray(decoded_img[0]).show()

rs.write_images_to_pkl(original_img, 'autoencoder_encoded.pkl')
rs.write_images_to_pkl(decoded_img, 'autoencoder_decoded.pkl')
