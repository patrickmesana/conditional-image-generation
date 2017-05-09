import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
import ntpath
import readsample as rs

def resize_mscoco():
    '''
    function used to create the dataset,
    Resize original MS_COCO Image into 64x64 images
    '''

    ### PATH need to be fixed
    data_path="/datasets/coco/coco/images/train2014"
    save_dir = "/Tmp/64_64/train2014/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preserve_ratio = True
    image_size = (64, 64)
    #crop_size = (32, 32)

    imgs = glob.glob(data_path+"/*.jpg")


    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print i, len(imgs), img_path

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

                ### Crop the 64/64 center
                tocrop = np.array(img)
                center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print tocrop.shape, center, (center[0]-32,center[0]+32), (center[1]-32,center[1]+32)
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                else:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))




def createInputAndTarget(
                  mscoco="inpainting/", split="train2014"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    imgs = glob.glob(data_path + "/*.jpg")

    for i, img_path in enumerate(imgs):
        print(img_path)
        img = Image.open(img_path)
        img_array = np.array(img)

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
        else:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]


        sample_name = 'training'
        if split == 'val2014':
            sample_name = 'validation'

        head, tail = ntpath.split(img_path)
        img_name = tail or ntpath.basename(head)

        Image.fromarray(img_array).save('./{sn}/target-full/{n}'.format(sn=sample_name, n=img_name))
        Image.fromarray(input).save('./{sn}/input/{n}'.format(sn=sample_name, n=img_name))
        Image.fromarray(target).save('./{sn}/target/{n}'.format(sn=sample_name, n=img_name))



def create_full_with_middle(imgs):
    n = len(imgs)
    np_images = []
    for i in range(0, n):
        img_array = imgs[i]
        black_img = np.zeros([64, 64, 3], dtype=np.uint8)
        center = (int(np.floor(black_img.shape[0] / 2.)), int(np.floor(black_img.shape[1] / 2.)))
        black_img[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = img_array
        np_images.append(img_array)
    return np_images

def persist_samples_on_disk():
    training_input_images = rs.read_sample_as_tensor(base_path="training/", split="input")
    training_target_full_images = rs.read_sample_as_tensor(base_path="training/", split="target-full")
    training_target_images = rs.read_sample_as_tensor(base_path="training/", split="target")
    validation_input_images = rs.read_sample_as_tensor(base_path="validation/", split="input")
    validation_target_full_images = rs.read_sample_as_tensor(base_path="validation/", split="target-full")
    validation_target_images = rs.read_sample_as_tensor(base_path="validation/", split="target")
    rs.write_images_to_pkl(training_input_images, 'training_input.pkl')
    rs.write_images_to_pkl(training_target_full_images, 'training_target_full.pkl')
    rs.write_images_to_pkl(training_target_images, 'training_target.pkl')
    rs.write_images_to_pkl(validation_input_images, 'validation_input.pkl')
    rs.write_images_to_pkl(validation_target_full_images, 'validation_target_full.pkl')
    rs.write_images_to_pkl(validation_target_images, 'validation_target.pkl')

# resize_mscoco()
# show_examples(5, 5, mscoco="/Users/patrickmesana/Dev/inpainting")
createInputAndTarget(split="train2014")
createInputAndTarget(split="val2014")
persist_samples_on_disk()

# rs.write_images_to_pkl(create_full_with_middle(x_train_target), 'training_target2.pkl')
# saved_images1 = rs.read_images_from_pkl('training_target2.pkl')
# rs.write_images_to_pkl(create_full_with_middle(x_test_target), 'validation_target2.pkl')
# saved_images2 = rs.read_images_from_pkl('validation_target2.pkl')