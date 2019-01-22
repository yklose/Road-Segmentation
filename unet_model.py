#------------- U-Net MODEL --------------

# As described in the report, this is the final model,
# used for the submissions. This model has more than 30 million parameters,
# and it is thus not advised to train it on a regular laptop as the memory
# usage might be too big to handle. It is also advised to run this on
# a GPU to speed up computation, as the training can take a long time.

# This file creates, trains and saves the model.
# To make predictions using this model, use the run.py file.

import numpy as np
import tensorflow as tf
import glob
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dropout, LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K
from jaccard_distance_loss import jaccard_distance_loss

# Define some global variables
IMG_SIZE = 400
NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
SEED = 1998
BATCH_SIZE = 8
NUM_EPOCHS = 100
STEPS_PER_EPOCH = 1000
RESTORE_MODEL = False # If True, restore existing model instead of training a new one


# Set the random seeds for reproducibility
np.random.seed(SEED)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def main(argv=None):

    # Load the training images and groundtruth. set x = True for images, and x = False for groundtruth
    def load_images(path, x=True):
        ids = glob.glob(path + "*.png")
        N = len(ids)
        images = np.empty((len(ids), IMG_SIZE, IMG_SIZE, NUM_CHANNELS if x else 1))
        for i, id_ in enumerate(ids):
            if (x):
                img = load_img(id_, target_size=(400,400))
            else:
                img = load_img(id_, target_size=(400,400), color_mode="grayscale")
            images[i] = img_to_array(img)/255.
        return images

    # Filepaths fot the training images and groundtruth
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    print("Loading images")
    # Extract it into np arrays.
    data = load_images(train_data_filename)
    labels = load_images(train_labels_filename, x=False)

    # Compute the f1-score for a given prediction.
    def f1_score(y_true, y_pred):
        # Count positive samples.
        c1 = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        c2 = np.sum(np.round(np.clip(y_pred, 0, 1)))
        c3 = np.sum(np.round(np.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0
        # How many selected items are relevant?
        precision = c1 / c2
        # How many relevant items are selected?
        recall = c1 / c3
        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, recall, precision

    # Create a metric to be able to compute the f1-score of the training set after each epoch
    # for better visualization.
    class Metrics(Callback):
      
      def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
      def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = np.array(self.validation_data[1])
        _val_f1, _val_recall, _val_precision = f1_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
        return

    metrics = Metrics()

    # Create a generator that rotates and flips the training images
    def train_generator(generator, x, y):
      x_generator = generator.flow(x, seed=SEED, batch_size=BATCH_SIZE)
      y_generator = generator.flow(y, seed=SEED, batch_size=BATCH_SIZE)
      while True:
        xi = x_generator.next()
        yi = y_generator.next()
        yield xi, yi

    generator = ImageDataGenerator(rotation_range=360.,
                       horizontal_flip = True)
    train_gen = train_generator(generator,data,labels)

    # Unet Model.
    def unet_model(data, n_filters=16, dropout=0.5):

        # A convolution block consisting of two layers of convolution + batchnormalization + activation (leaky relu)
        def conv_block(input_tensor, n_filters, kernel_size=3):
            # first layer
            x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_uniform",
                       padding="same")(input_tensor)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            # second layer
            x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="random_uniform",
                       padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            return x

        # 5-layer contracting path with maxpoolings in between and dropout for regularization.
        c1 = conv_block(input_img, n_filters=n_filters*1, kernel_size=5)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = conv_block(p1, n_filters=n_filters*2, kernel_size=3)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv_block(p2, n_filters=n_filters*4, kernel_size=3)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv_block(p3, n_filters=n_filters*8, kernel_size=3)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = conv_block(p4, n_filters=n_filters*16, kernel_size=3)
        
        # 4-layer expansive path with transpose convolutions
        u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4])
        c6 = conv_block(u6, n_filters=n_filters*8, kernel_size=3)

        u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3])
        c7 = conv_block(u7, n_filters=n_filters*4, kernel_size=3)

        u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2])
        c8 = conv_block(u8, n_filters=n_filters*2, kernel_size=3)

        u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = conv_block(u9, n_filters=n_filters*1, kernel_size=3)
        
        # 1 x 1 convolution to get an output shape of (400 x 400 x 1)
        output = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[input_img], outputs=[output])
        return model

    # If restore model, restore an existing model to continue training.
    # Change the model filename to the correct one as needed.
    if RESTORE_MODEL:
        print("Restoring model")
        model_filename = "unet_model_v1.h5"
        model = load_model(model_filename, custom_objects={'jaccard_distance_loss': jaccard_distance_loss})
    else:
        print("Creating model")
        input_img = Input((IMG_SIZE,IMG_SIZE,NUM_CHANNELS), name='image')
        model = unet_model(input_img, n_filters=64)
        model.compile(optimizer=Adam(), loss=jaccard_distance_loss)
        model.summary()

    # train the model for a given number of epoch and a given amount of steps per epochs.
    print("Training model for {} epochs, with {} steps per epoch".format(NUM_EPOCHS, STEPS_PER_EPOCH))
    model.fit_generator(train_gen, epochs=NUM_EPOCHS, validation_data=(data, labels), steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[metrics])

    # Save the trained model
    print("Saving trained model")
    new_model_filename = "unet_model_v1.h5"
    model.save(new_model_filename)

if __name__ == '__main__':
    tf.app.run()
