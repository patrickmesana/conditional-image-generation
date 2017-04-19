
rning Class Inpainting dataset, http://ift6266h17.wordpress.com/ *


** Dataset

The inpainting dataset is a downsample version of the MSCOCO dataset (http://mscoco.org/dataset/#overview)

The original images in the MSCOCO dataset are high resolution: roughly 500×500 pixels.  While this will give the most interesting results, it would likely be too computationally intensive for a class project (and indeed, for most research projects).  Many successful generative modeling papers still evaluate on 32×32 images.  In fact, there can still be a lot of detail in a 32×32 image!



Some of the images might be in grayscale instead of RGB, you can just skip those images.


** Files

In particular the archive inpainting.tar.bz2 contains:
- train2014: a directory composed by 82782 training images
- val2014:a directory composed by 40504 validations images
- dict_key_imgID_value_caps_train_and_valid.pkl: a pickled python dictionary containing the captions associated to the train/valid images
- worddict.pkl: a pickled dictionary of the different words composing the captions.

To extract the archive:
tar xjvf inpainting.tar.bz2


** Examples

We also provides the python script examples.py
Code used to downsample the images, and a function to visualize the batch are in examples.py
We also show how to construct the input/target from the images and how to visualize the datas,


Please refer to the class website, http://ift6266h17.wordpress.com/project-description/, for more details.
