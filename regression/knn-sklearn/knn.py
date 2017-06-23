# Regression task - knn by scikit-learn

# @HYPJUDY 2017.6.21
# Details: https://hypjudy.github.io/2017/06/23/regression-classification-kaggle/

import tensorflow as tf
import numpy as np
import csv
import os
import time
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
# Define paramaters for the model
TRAIN_PATH = 'save_train.csv'
TEST_PATH = 'save_test.csv'

BATCH_SIZE = 25000 # test data and train data both have 25000 items
N_FEATURES = 384

def train_batch_generator(filenames):
    """ filenames is the list of files you want to read from. 
        In this case, it contains only save_train.csv
    """
    # Read in files from queues
    filename_queue = tf.train.string_input_producer(filenames)
    # Outputs the lines of a file delimited by newlines. E.g. text files, CSV files
    reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
    _, value = reader.read(filename_queue)

    # record_defaults are the default values in case some of our columns are empty
    # This is also to tell tensorflow the format of our data (the type of the decode result)
    record_defaults = [[0.0] for _ in range(N_FEATURES + 2)] # id + 384 values + reference

    # read in the all rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults)

    data_batch = content[1:N_FEATURES+1]

    # assign the last column to label
    label_batch = content[-1]
    
    return data_batch, label_batch

def train_generate_batches(data_batch, label_batch):
    features = [([0.0]*N_FEATURES) for _ in range(BATCH_SIZE)]
    labels = [[0.0] for _ in range(BATCH_SIZE)]
    with tf.Session() as sess:
        # start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(BATCH_SIZE): # generate batches
            features[i], labels[i][0] = sess.run([data_batch, label_batch])
        coord.request_stop()
        coord.join(threads)
    return features, labels


def test_batch_generator(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(filename_queue)

    record_defaults = [[0.0] for _ in range(N_FEATURES + 1)] # id + 384 values

    content = tf.decode_csv(value, record_defaults=record_defaults)

    data_batch = content[1:N_FEATURES+1]

    return data_batch

def test_generate_batches(data_batch):
    # test data do not have labels
    features = [([0.0]*N_FEATURES) for _ in range(BATCH_SIZE)]
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(BATCH_SIZE):
            features[i] = sess.run(data_batch)
        coord.request_stop()
        coord.join(threads)
    return features


def main():
    s = time.time()
    # Step 1: Read data
    print 'Reading training data...'
    train_data_batch, train_label_batch = train_batch_generator([TRAIN_PATH])
    train_features, train_labels = train_generate_batches(train_data_batch, train_label_batch)


    print 'Reading testing data...'
    test_data_batch = test_batch_generator([TEST_PATH])
    test_features = test_generate_batches(test_data_batch)
    
    e = time.time()
    print "Reading time:", (e - s), "seconds."

    s = time.time()
    # Step 2: Define model
    knn = neighbors.KNeighborsRegressor(n_neighbors=1)
    
    ## Test by spliting data into new training data and testing data 
    # X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.3, random_state=0)
    # print knn.fit(X_train, y_train).score(X_test, y_test)
    
    ## Cross validation
    # scores = cross_val_score(knn, train_features, train_labels, cv=10, scoring='neg_mean_squared_error')
    # print (scores, np.sqrt(np.abs(scores.mean())))

    # Step 3: Fit the model and predict
    test_labels = knn.fit(train_features, train_labels).predict(test_features)
    with open('res_nn.csv', 'w') as fw:
        csv_w = csv.writer(fw)
        csv_w.writerow(['Id', 'reference'])
        for i in range(BATCH_SIZE):
            csv_w.writerow([i, test_labels[i][0]])
    e = time.time()
    print "Training and testing time:", (e - s), "seconds."


if __name__ == '__main__':
    main()
