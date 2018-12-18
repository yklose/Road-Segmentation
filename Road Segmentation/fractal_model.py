#------------- FRACTAL MODEL --------------

# As described in the report, this model was our first attempt 
# to improve the model. We did not used this model for our 
# final result, but used it instead to compare our final model
# with another neural network architechture.


import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

from sklearn.metrics import f1_score

import code

import tensorflow.python.platform

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
import matplotlib.pyplot as plt

NUM_CHANNELS = 3                # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
VALIDATION_SIZE = 5             # Size of the validation set.
SEED = 1998                     # Set to None for random seed.
BATCH_SIZE = 16                 # 64
NUM_EPOCHS = 50
RESTORE_MODEL = True            # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

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

    return numpy.asarray(data)
        
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
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

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

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS

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
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
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

    # Get prediction for given input image 
    def get_prediction(img):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        output_prediction = model.predict(data)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
        return img_prediction
    
    # Compute Advanced prediction based on given image prediction
    def advanced_prediction(data):
        
        #the borders will not be taken into account
        #because roads might end there or are cutted
        #diagonal elements are checked as well     
        
        for i in range(1,24):
            for j in range(1, 24):
                if (data[i*16, j*16] == 0 and (data[16*(i-1), j*16] + data[16*(i+1), j*16] + data[16*i, 16*(j-1)] + data[16*i, 16*(j+1)] + data[16*(i-1), (j-1)*16] + data[16*(i-1), (j+1)*16] + data[16*(i+1), (j-1)*16] + data[16*(i+1), (j+1)*16] == 8)):
                    #print("0 -> 1")
                    data[i*16:(i+1)*16, j*16:(j+1)*16] = 1
                elif (data[i*16, j*16] == 1 and (data[16*(i-1), j*16] + data[16*(i+1), j*16] + data[16*i, 16*(j-1)] + data[16*i, 16*(j+1)] + data[16*(i-1), (j-1)*16] + data[16*(i-1), (j+1)*16] + data[16*(i+1), (j-1)*16] + data[16*(i+1), (j+1)*16] == 0)):
                    #print("1 -> 0")
                    data[i*16:(i+1)*16, j*16:(j+1)*16] = 0
        
 
        return data

    # Compute the accuracy
    def compute_accuracy(img_prediction, y_true):
            n = len(img_prediction)
            buffer = 0
            
            for i in range(n):
                if (img_prediction[i]-y_true[i] == 0):
                    buffer += 1
            
            accuracy = buffer/n
            print("Accuracy on Image: ", str(accuracy))
            
            
            return accuracy
        
    # Compute the f1-score
    def compute_f1_score(img, image_idx):
        
        #this functions also calls the compute_accuary
        #the accuracy is, as mentioned in report however not as relevant as f1-score
        
        # f1 score prediction before 
        img_prediction = get_prediction(img)
        y_true = 1 - load_y_true(image_idx)
        y_true = y_true.flatten()
        img_prediction = img_prediction.flatten()
        a_before = compute_accuracy(img_prediction, y_true)
        f1_before = f1_score(y_true, img_prediction, average='macro')
        # reshape
        y_true = y_true.reshape((400, 400))
        img_prediction = img_prediction.reshape((400,400))

        # f1 score prediction after
        img_prediction = advanced_prediction(img_prediction)
        y_true = y_true.flatten()
        img_prediction = img_prediction.flatten()
        a_after = compute_accuracy(img_prediction, y_true)
        f1_after = f1_score(y_true, img_prediction, average='macro')
        # reshape for cimg
        y_true = y_true.reshape((400, 400))
        img_prediction = img_prediction.reshape((400,400))
        
        f1_after = f1_before
        
        return img_prediction, f1_before, f1_after, a_before, a_after

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        
        img_prediction, f1_before, f1_after, a_before, a_after = compute_f1_score(img, image_idx)
        
        cimg = concatenate_images(img, img_prediction)
        
        return cimg, f1_before, f1_after, a_before, a_after

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        #img_prediction = get_prediction(img)
        
        img_prediction, _, _, _, _ = compute_f1_score(img, image_idx)
        
        oimg = make_img_overlay(img, img_prediction)

        return oimg
    
    # Load the y_true image
    def load_y_true(image_idx):
        imageid = "satImage_%.3d" % image_idx
        image_filename = train_labels_filename + imageid + ".png"
        img = mpimg.imread(image_filename)
        
        return numpy.rint(img)
    
    # Fractal Model.
    def fract_model(data, filters=16):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].

        def convBlock(data, filters, batchnorm=True, activation='relu'):
          conv = Conv2D(filters=filters, kernel_size=(3,3), padding='same')(data)
          bn = BatchNormalization()(conv)
          act = LeakyReLU(alpha=0.1)(bn)
          return act

        def fract_conv(data, filters, depth=1):
            if (depth<=1):
                return convBlock(data, filters, batchnorm=True, activation='relu')
            else:
                conv = convBlock(data, filters, batchnorm=True, activation='relu')
                conv2 = fract_conv(data, filters, depth=depth-1)
                conv3 = fract_conv(conv2, filters, depth=depth-1)
                conc = concatenate([conv,conv3])
                return conc

        dropout=0.5

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

    if RESTORE_MODEL:
        model = load_model("fract_model_16p_16f.h5")
    else:
        input_patch = Input((IMG_PATCH_SIZE,IMG_PATCH_SIZE,3), name='patch')
        model = fract_model(input_patch, filters = 8)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=["accuracy"])
        model.summary()

        #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(train_data, train_labels, validation_split=0.25, epochs=num_epochs, batch_size=BATCH_SIZE, verbose=1)
        model.save("fract_model_16p_8f.h5")
        
        del model
        
        model = fract_model(input_patch, filters = 32)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=["accuracy"])
        model.summary()

        #early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        history = model.fit(train_data, train_labels, validation_split=0.25, epochs=num_epochs, batch_size=BATCH_SIZE, verbose=1)
        model.save("fract_model_16p_32f.h5")

    print ("Running prediction on training set")
    prediction_training_dir = "predictions_training/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)
    f1_array_before = []
    f1_array_after = []
    a_array_before = []
    a_array_after = []
    for i in range(1, TRAINING_SIZE+1):
        pimg, img_f1_score_before, img_f1_score_after, a_before, a_after  = get_prediction_with_groundtruth(train_data_filename, i)
        Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
        oimg = get_prediction_with_overlay(train_data_filename, i)
        oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        f1_array_before.append(img_f1_score_before)
        f1_array_after.append(img_f1_score_after)
        a_array_before.append(a_before)
        a_array_after.append(a_after)
    print("------- Average F1 Score -------")
    print("normal F1          : " + str(numpy.mean(f1_array_before)))
    print("normal Accuracy    : " + str(numpy.mean(a_array_before)))
    print("advanced  F1       : " + str(numpy.mean(f1_array_after)))
    print("advanced Accuracy  : " + str(numpy.mean(a_array_after)))

if __name__ == '__main__':
    tf.app.run()
