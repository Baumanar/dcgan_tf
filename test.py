import model_cifar
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
n_noise = 64

tf.reset_default_graph()


with tf.Session() as sess:
    gan = model_cifar.Gan()
    saver = tf.train.Saver()
    saver.restore(sess, 'log/cifar_sig_-599')

    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
    feed_dict = {
                 gan.generator_input: n,
                 gan.keep_prob: 1,
                 gan.is_training: False}

    test_img = sess.run(gan.generator, feed_dict=feed_dict)
    print(test_img.shape)



fig=plt.figure(figsize=(8, 8))
columns = 8
rows = 8
for i in range(1, columns*rows +1):
    img =test_img[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(np.reshape(img,(32,32,3)))
plt.show()
