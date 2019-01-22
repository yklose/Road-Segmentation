## Machine Learning Project
Road Segmentation on satellite imagery

Group: Maxime Schoemans, Pedro de Tavora Santos and Yannick Paul Klose

CrowdAI team name and submission ID: mschoema // 23930

### Structure of Code
The code is structed in 5 different components, that are represented by the following python files:

##### "basic_convolutional_model.py"
This was the first model implemented by the group and it is heavily based on the code given to us by the Machine Learning course team. This model uses a two-layer convolutional neural network in order to make the predictions.


##### "fractal_model.py"
This file implements our first real optimization of the aforementioned model. It uses a fractal architecture with 4-layer structure in order to make the predictions.

##### "unet_model.py"
This file implements the final iteration of our models: it creates a u-net architecture with a 9 layer architecture, trains it and saves it in a file, to be later used in the run.py file for the prediction on the test set.

##### "jaccard_distance_loss.py"
This file implements the Jaccard distance loss that is used to compute the loss that is used to train out u-net model.

##### "run.py"
This is the file that we use to make the final submission file based on the u-net model.

Additionally, we have 2 folders: one containing the training images (satellite images and groundtruth) and the other containing the test images used to make the submission file.
##########################################
#How to run the code
Since our model has close to 35 million parameters it is impossible to train within a reasonable time on a regular computer without the use of external GPU's.
So the first step is to download our model file from the google drive link: https://drive.google.com/open?id=17Zh2KGauj2v3iP-04RdJkguKtvM6O-em .
and save it in the same folder as the run.py. After you simply run the run.py file and the output will be the aforementioned submission file.

#Warnings
The load_model function will take up a lot of memory space and it takes some time before the submission file is made.
