# Regression task - Multi-Layer Perceptron by TensorFlow (short version)
#
# @HYPJUDY 2017.6.21
# Details: https://hypjudy.github.io/2017/06/23/regression-classification-kaggle/

import tensorflow as tf
import numpy as np
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# Define paramaters for the model
TRAIN_PATH = 'save_train.csv'
TEST_PATH = 'save_test.csv'
RESULT_PATH = 'submission'
BATCH_SIZE = 25000
N_FEATURES = 384
LEARNING_RATE = 0.001
NODE_NUM = 100
MAX_STEP = 30000
SAVE_ITERS = 1000

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

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

def main():
    # Step 1: Read in data
    print 'Reading training data...'
    train_data_batch, train_label_batch = train_batch_generator([TRAIN_PATH])
    train_features, train_labels = train_generate_batches(train_data_batch, train_label_batch)
    print np.shape(train_features)
    print np.shape(train_labels)

    print 'Reading testing data...'
    test_data_batch = test_batch_generator([TEST_PATH])
    test_features = test_generate_batches(test_data_batch)
    print np.shape(test_features)


    # Step 2: create placeholders for features and labels
    with tf.name_scope('data'):
        X = tf.placeholder(tf.float32, [None, 384], name="X_placeholder")
        Y = tf.placeholder(tf.float32, [None, 1], name="Y_placeholder")

    # Step 3: create weight and bias
    w1 = tf.Variable(xavier_init(384, NODE_NUM), name='weights1')
    b1 = tf.Variable(tf.constant(1.0, shape=[NODE_NUM], dtype=tf.float32), name='bias1')
    w2 = tf.Variable(xavier_init(NODE_NUM, NODE_NUM), name='weights2')
    b2 = tf.Variable(tf.constant(1.0, shape=[NODE_NUM], dtype=tf.float32), name='bias2')
    w3 = tf.Variable(xavier_init(NODE_NUM, NODE_NUM), name='weights3')
    b3 = tf.Variable(tf.constant(1.0, shape=[NODE_NUM], dtype=tf.float32), name='bias3')
    w4 = tf.Variable(xavier_init(NODE_NUM, NODE_NUM), name='weights4')
    b4 = tf.Variable(tf.constant(1.0, shape=[NODE_NUM], dtype=tf.float32), name='bias4')
    w5 = tf.Variable(xavier_init(NODE_NUM, 1), name='weights5')
    b5 = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32), name='bias5')

    # Step 4: build model to predict Y
    hidden = tf.sigmoid(tf.add(tf.matmul(X, w1), b1))
    hidden2 = tf.sigmoid(tf.add(tf.matmul(hidden, w2), b2))
    hidden3 = tf.sigmoid(tf.add(tf.matmul(hidden2, w3), b3))
    hidden4 = tf.sigmoid(tf.add(tf.matmul(hidden3, w4), b4))
    Y_predicted = tf.add(tf.matmul(hidden4, w5), b5)

    # Step 5: use the square error as the loss function
    loss = tf.reduce_mean(tf.square(Y - Y_predicted), name='loss')

    # Step 6: using gradient descent to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # Launch the graph and train
    with tf.Session() as sess:
        # Step 7: initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer()) 
    
        writer = tf.summary.FileWriter('./graph', sess.graph)
    
        # Step 8: train the model
        for step in xrange(MAX_STEP):
            # Session runs train_op and fetch values of loss
            print 'Training step ', step
            _, l = sess.run([optimizer, loss], feed_dict={X:train_features, Y:train_labels}) 
            print 'Step {0}: loss = {1}'.format(step, np.sqrt(l)) # loss <- RMSE
            
            # Step 9: Test the model every SAVE_ITERS steps
            if (step + 1) % SAVE_ITERS == 0:
                result = sess.run(Y_predicted, feed_dict={X:test_features})
                filename = RESULT_PATH + '_' + str(step) + '_' + str(int(np.sqrt(l)*100))+ '.csv'
                print filename
                with open(filename, 'w') as fw:
                    csv_w = csv.writer(fw)
                    csv_w.writerow(['Id', 'reference'])
                    for i in range(BATCH_SIZE):
                        csv_w.writerow([i, result[i][0]])

        # close the writer when you're done using it
        writer.close() 


if __name__ == '__main__':
    main()