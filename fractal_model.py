#------------- FRACTAL MODEL --------------

# As described in the report, this model was our first attempt 
# to improve the model. We did not used this model for our 
# final result, but used it instead to compare our final model
# with another neural network architechture.


import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import numpy
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Activation, BatchNormalization, Dropout, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Define some global variables
NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 1998  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 50
RESTORE_MODEL = True # If True, restore existing model instead of training a new one
ADVANCED_PREDICTION = True
RECORDING_STEP = 1000

# Set image patch size in pixels
IMG_PATCH_SIZE = 16

# Set the random seeds for reproducibility
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return imgs, numpy.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return numpy.rint(gt_imgs), labels.astype(numpy.float32)

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    print("Loading images")
    imgs, data = extract_data(train_data_filename, TRAINING_SIZE)
    gt_imgs, labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    if not RESTORE_MODEL:

        num_epochs = NUM_EPOCHS//2

        c0 = 0
        c1 = 0
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 = c0 + 1
            else:
                c1 = c1 + 1
        print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

        print ('Balancing training data...')
        min_c = min(c0, c1)
        idx0 = [i for i, j in enumerate(labels) if j[0] == 1]
        idx1 = [i for i, j in enumerate(labels) if j[1] == 1]
        numpy.random.shuffle(idx0)
        numpy.random.shuffle(idx1)
        new_indices = idx0[0:min_c] + idx1[0:min_c]
        print (len(new_indices))
        print (train_data.shape)
        train_data = train_data[new_indices,:,:,:]
        train_labels = train_labels[new_indices]


        train_size = train_labels.shape[0]

        c0 = 0
        c1 = 0
        for i in range(len(train_labels)):
            if train_labels[i][0] == 1:
                c0 = c0 + 1
            else:
                c1 = c1 + 1
        print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # Get prediction, f1 score before the advanced prediction and f1 score after, for a given input image
    def get_prediction(img, gt_image):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        output_prediction = model.predict(data)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
        f1_before = compute_f1_score(img_prediction, gt_image)
        if ADVANCED_PREDICTION:
            img_prediction = advanced_prediction(img_prediction)
            f1_after = compute_f1_score(img_prediction, gt_image)
        else:
            f1_after = None
        return img_prediction, f1_before, f1_after

    # compute the f1 score an image
    def compute_f1_score(img_prediction, y_true):
        y_true = 1 - y_true
        y_true = y_true.flatten()
        img_prediction = img_prediction.flatten()
        score = f1_score(y_true, img_prediction, average='macro')
        return score

    # Remove isolated patches in the predicted image.
    def advanced_prediction(data):
        #the borders will not be taken into account
        #because roads might end there or are cutted
        #diagonal elements are checked as well        
        for i in range(1,24):
            for j in range(1, 24):
                if (data[i*16, j*16] == 0 and (data[16*(i-1), j*16] + data[16*(i+1), j*16] + data[16*i, 16*(j-1)] + data[16*i, 16*(j+1)] + data[16*(i-1), (j-1)*16] + data[16*(i-1), (j+1)*16] + data[16*(i+1), (j-1)*16] + data[16*(i+1), (j+1)*16] == 8)):
                    data[i*16:(i+1)*16, j*16:(j+1)*16] = 1
                elif (data[i*16, j*16] == 1 and (data[16*(i-1), j*16] + data[16*(i+1), j*16] + data[16*i, 16*(j-1)] + data[16*i, 16*(j+1)] + data[16*(i-1), (j-1)*16] + data[16*(i-1), (j+1)*16] + data[16*(i+1), (j-1)*16] + data[16*(i+1), (j+1)*16] == 0)):
                    data[i*16:(i+1)*16, j*16:(j+1)*16] = 0
        return data

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(img, img_prediction):
        cimg = concatenate_images(img, img_prediction)
        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(img, img_prediction):
        oimg = make_img_overlay(img, img_prediction)
        return oimg

    # Fractal Model.
    def fract_model(data, filters=16):

        # Convolutional block with a convolution followed by a batch normalization and a leaky relu activation
        def convBlock(data, filters, batchnorm=True, activation='relu'):
          conv = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(data)
          bn = BatchNormalization()(conv)
          act = LeakyReLU(alpha=0.1)(bn)
          return act

        # Create a fractal scruture in the model, like described on the report
        def fract_conv(data, filters, depth=1):
            if (depth<=1):
                return convBlock(data, filters, batchnorm=True, activation='relu')
            else:
                conv = convBlock(data, filters, batchnorm=True, activation='relu')
                conv2 = fract_conv(data, filters, depth=depth-1)
                conv3 = fract_conv(conv2, filters, depth=depth-1)
                conc = concatenate([conv,conv3])
                return conc

        # Use of dropout for regularization
        dropout=0.5

        # 4 layer sctructure to go from a (16 x 16 x 3) input shape to (1 x 1 x 1)
        conv = fract_conv(data, filters=filters, depth=3)
        pool = MaxPooling2D((2,2))(conv)
        dr = Dropout(dropout)(pool)
        conv2 = fract_conv(dr, filters=filters*2, depth=3)
        pool2 = MaxPooling2D((2,2))(conv2)
        dr2 = Dropout(dropout)(pool2)
        conv3 = fract_conv(dr2, filters=filters*4, depth=3)
        pool3 = MaxPooling2D((2,2))(conv3)
        dr3 = Dropout(dropout)(pool3)
        conv4 = fract_conv(dr3, filters=filters*8, depth=3)
        pool4 = MaxPooling2D((2,2))(conv4)
        dr4 = Dropout(dropout)(pool4)
        conv5 = Conv2D(2,(1,1), activation='sigmoid')(dr4)
        outputs = Flatten()(conv5)

        model = Model(inputs=[data], outputs=[outputs])
        return model

    # If restore model, restore an existing model to make predictions.
    # Change the model filename to the correct one as needed.
    if RESTORE_MODEL:
        print("Restoring model")
        model_filename = "fract_model_v1.h5"
        model = load_model(model_filename)
    else:
        input_patch = Input((IMG_PATCH_SIZE,IMG_PATCH_SIZE,3), name='patch')
        model = fract_model(input_patch)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=["accuracy"])
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(train_data, train_labels, validation_split=0.25, epochs=num_epochs, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stopping])
        new_model_filename = "fract_model_v2.h5"
        model.save(new_model_filename)

    # Run the prediction on the training set, compute average f1 score before and after the advanced prediction
    # and create images with a prediction overlay and groundtruth images after advanced prediction. (the advanced prediction is optional)

    print ("Running prediction on training set")
    prediction_training_dir = "predictions_training/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)

    f1_array_before = []
    if ADVANCED_PREDICTION:
        f1_array_after = []
    for i in range(1, TRAINING_SIZE+1):
        img_prediction, f1_before, f1_after = get_prediction(imgs[i-1], gt_imgs[i-1])
        f1_array_before.append(f1_before)
        if ADVANCED_PREDICTION:
            f1_array_after.append(f1_after)
        pimg = get_prediction_with_groundtruth(imgs[i-1],img_prediction)
        Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(imgs[i-1],img_prediction)
        oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
    print("------- Average F1 Score -------")
    print("Without advanced prediction : " + str(numpy.mean(f1_array_before)))
    if ADVANCED_PREDICTION:
        print("With advanced prediction  : " + str(numpy.mean(f1_array_after)))

if __name__ == '__main__':
    tf.app.run()
