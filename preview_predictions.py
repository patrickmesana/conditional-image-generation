import readsample as rs
import numpy as np
import display


def preview_full(name):
    x_test_input_unchanged = rs.read_images_from_pkl('validation_input.pkl')
    imgs = rs.read_images_from_pkl(name)

    n = 5
    np_images = []
    for i in range(0, n):
        img_array = imgs[i]
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        np_images.append(target)

    display.show_reconstructed_images(x_test_input_unchanged, np_images, True)



def preview_middle(name):
    x_test_input_unchanged = rs.read_images_from_pkl('validation_input.pkl')
    reshaped_decoded_imgs = rs.read_images_from_pkl(name)


    display.show_reconstructed_images(x_test_input_unchanged, reshaped_decoded_imgs, True)


def preview_predictions(name):
    imgs = rs.read_images_from_pkl(name)
    display.show_images(imgs)

# preview_middle('autoencoder_decoded.pkl')
#preview_full('gan_decoded.pkl')
preview_predictions('convolution_decoded.pkl')