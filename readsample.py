import os
import glob
import numpy as np
import PIL.Image as Image
import pickle

def read_sample_as_tensor(base_path="training/", split="target"):
    '''
    read sample from training sample or validation
    '''

    data_path = os.path.join(base_path, split)

    imgs = glob.glob(data_path + "/*.jpg")

    np_images = []
    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_array = np.array(img)
        # we only keep colored images
        if len(img_array.shape) == 3:
            np_images.append(img_array)
    return np.asarray(np_images)


def write_images_to_pkl(np_images, path):
    with open(path, 'wb') as output:
        pickle.dump(np_images, output, pickle.HIGHEST_PROTOCOL)

def read_images_from_pkl(path):
    with open(path, 'rb') as input:
        images = pickle.load(input)
    return images