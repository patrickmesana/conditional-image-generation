# if you want to run this on GPU
#THEANO_FLAGS=device=gpu,floatX=float32 python autoencoder.py


from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import readsample as rs
import PIL.Image as Image

img_resolution = 64*64*3 # 12,288
# this is the size of our encoded representations
encoding_dim = img_resolution # compression factor of 2
# this is our input placeholder (we want to keep a reference to this, although it's part of the Model)
input_img = Input(shape=(img_resolution,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(img_resolution, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

x_train = rs.read_images_from_pkl('training_target.pkl')
x_test = rs.read_images_from_pkl('validation_target.pkl')[0:100]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

original_img = x_test[0].reshape(64, 64, 3) * 255.
original_img = original_img.astype('uint8')
decoded_img = decoded_imgs[0].reshape(64, 64, 3) * 255.
decoded_img = decoded_img.astype('uint8')
Image.fromarray(original_img).show()
Image.fromarray(decoded_img).show()
