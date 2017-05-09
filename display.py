import PIL.Image as Image
import numpy as np
import os, sys
import glob
import cPickle as pkl

def show_reconstructed_images(inputs, outputs, saveImageOnDisk, n = 5):
    for i in range(0, n):
        img_array = np.copy(inputs[i])
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = outputs[i]
        Image.fromarray(img_array).show()
        if saveImageOnDisk: Image.fromarray(img_array).save('PredictedImage{i}.jpg'.format(i=i))

def show_images(inputs, n = 5, saveImageOnDisk= False):
    for i in range(0, n):
        if saveImageOnDisk:
            Image.fromarray(inputs[i]).save('PredictedImage{i}.jpg'.format(i=i))
        else:
            Image.fromarray(inputs[i]).show()

def show_examples(batch_idx, batch_size,
                  ### PATH need to be fixed
                  mscoco="inpainting/", split="train2014",
                  caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path) as fd:
        caption_dict = pkl.load(fd)

    print data_path + "/*.jpg"
    imgs = glob.glob(data_path + "/*.jpg")

    # if batch_idx is 5 and batch_size is 5 then we go at 5x5 index and take the 5 from there
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]

        # Image.fromarray(img_array).show()
        Image.fromarray(input).show()
        Image.fromarray(target).show()
        print img_path
        print i, caption_dict[cap_id]