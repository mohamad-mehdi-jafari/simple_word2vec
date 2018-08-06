# necessary libraries
import tensorflow as tf
import random
import numpy as np


# import utils file
import utils
import numpy as np

import time
# sampling


# configuration parameters
VOCABULARY_SIZE = 63232
EMBEDDING_DIMENSIONALITY = 50
WINDOW_SIZE = 3
NEGATIVE_SAMPLING = 3
LEARNING_RATE = 0.1
BATCH_SIZE = 128
NUM_TRAIN_STEPS = 100
SKIP_STEP = 10



# important path inside machine
path1 = "clean_file.txt"
path2 = "dictionary"


batch_generator = utils.batch_generator(batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, negative_sample_count=NEGATIVE_SAMPLING,
                                    clear_text= path1)

def gen():
    while True:
        batch = next(batch_generator)
        batch = np.array(batch, dtype=np.int32)
        centers = np.reshape(batch[:,0], newshape=(BATCH_SIZE))
        targets = np.reshape(batch[:,1], newshape=(BATCH_SIZE, 1))

        yield centers, targets



def graph(dataset):
    iterator = dataset.make_initializable_iterator()
    center_words, target_words =  iterator.get_next()

    '''
        pay attention that we don't need any place holder in this model
        cause we have our batch lables already
    '''

    # define weights.
    # In word2vec, it's the weights that we care about
    embed_matrix = tf.get_variable('embed_matrix',
                                   shape=[VOCABULARY_SIZE, EMBEDDING_DIMENSIONALITY],
                                   initializer=tf.random_uniform_initializer())

    # define lookup table using weight matrix and our indecis
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')


    # construct variables for NCE loss, this part only adds freedom degree to our model
    nce_weight = tf.get_variable('nce_weight',
                                 shape=[VOCABULARY_SIZE, EMBEDDING_DIMENSIONALITY],
                                 initializer=tf.truncated_normal_initializer(stddev=1.0 / (EMBEDDING_DIMENSIONALITY ** 0.5)))
    nce_bias = tf.get_variable('nce_bias', initializer=tf.zeros([VOCABULARY_SIZE]))

    # define loss function to be NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                         biases=nce_bias,
                                         labels=target_words,
                                         inputs=embed,
                                         num_sampled=NEGATIVE_SAMPLING,
                                         num_classes=VOCABULARY_SIZE), name='loss')


    # Gradient Descend :  to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    #

    # define saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Step 6: initialize iterator and variables
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        # print(sess.run(center_words))

        total_loss = 0.0  # we use this to calculate late average loss in the last SKIP_STEP steps

        #
        for index in range(NUM_TRAIN_STEPS):
            try:

                loss_batch, _ = sess.run([loss, optimizer])

                total_loss += loss_batch

                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)

        saver.save(sess=sess, save_path="")



if __name__ == "__main__":
    dataset = tf.data.Dataset.from_generator(gen,
                                    (tf.int32, tf.int32),
                                    (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))


    graph(dataset)



