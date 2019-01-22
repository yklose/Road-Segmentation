#------------- Create a submission file --------------

# This file uses a pretrained unet model. Please download
# a our pretrained model from https://drive.google.com/open?id=17Zh2KGauj2v3iP-04RdJkguKtvM6O-em
# To create, train and save a new model, use the unet_model.py file.

import numpy as np
import matplotlib.image as mpimg
from keras.models import load_model
from jaccard_distance_loss import jaccard_distance_loss

# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
# The test images are 608 x 608 pixels, but the model input is 400 x 400,
# so we create 4 new images from the test image, compute a prediction for these four images
# and merge them back together to end up with the prediction of the initial test image.
def predict_img(img):
    img1 = img[:400,:400]
    img2 = img[:400,-400:]
    img3 = img[-400:,:400]
    img4 = img[-400:,-400:]
    imgs = np.array([img1,img2,img3,img4])
    labels = model.predict(imgs).round()
    img_label = np.empty((608,608,1))
    img_label[-400:,-400:] = labels[3]
    img_label[-400:,:400] = labels[2]
    img_label[:400,-400:] = labels[1]
    img_label[:400,:400] = labels[0]
    return img_label
    
# Creates the prediction for a single image and outputs the strings that should go into the submission file
def mask_to_submission_strings(image_filename, i):
    img_number = i+1
    print("Predicting image {}".format(image_filename))
    im = mpimg.imread(image_filename)
    im_label = predict_img(im)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im_label[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

# Converts images into a submission file
def masks_to_submission(submission_filename, *image_filenames):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(image_filenames)):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_filenames[i],i))


if __name__ == '__main__':

    # Restores the trained unet. Download it from https://drive.google.com/open?id=17Zh2KGauj2v3iP-04RdJkguKtvM6O-em,
    # save it in the same file as run.py and make sure the "model_filename" is correct.
    print("Restoring model")
    model_filename = "unet_model_v3.h5"
    model = load_model(model_filename, custom_objects={'jaccard_distance_loss': jaccard_distance_loss})

    # Creates a submission file in the correct format.
    print("Making submission file")
    submission_filename = 'submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'test_set_images/test_' + '%d' % i +'/test_' + '%d' % i + '.png'
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)

