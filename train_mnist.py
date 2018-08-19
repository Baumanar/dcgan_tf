import model_mnist
from data_utils import *
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
n_noise = 64



tf.reset_default_graph()


with tf.Session() as sess:
    gan = model_mnist.Gan()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(60000):
        train_d = True
        train_g = True
        keep_prob_train = 0.5  # 0.5

        n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
        batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]

        # batch = next_batch(images_norm, 64)
        feed_dict = {gan.discriminator_input: batch,
                     gan.generator_input: n,
                     gan.keep_prob: keep_prob_train,
                     gan.is_training: True}

        d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([gan.loss_d_real, gan.loss_d_fake, gan.loss_g, gan.loss_d],
                                                    feed_dict=feed_dict)

        d_real_ls = np.mean(d_real_ls)
        d_fake_ls = np.mean(d_fake_ls)
        g_ls = g_ls
        d_ls = d_ls

        if g_ls * 1.5 < d_ls:
            train_g = False
            pass
        if d_ls * 2 < g_ls:
            train_d = False
            pass

        if train_d:
            sess.run(gan.optimizer_d, feed_dict={gan.generator_input: n,
                                                 gan.discriminator_input: batch,
                                                 gan.keep_prob: keep_prob_train,
                                                 gan.is_training: True})

        if train_g:
            sess.run(gan.optimizer_g, feed_dict={gan.generator_input: n,
                                                 gan.keep_prob: keep_prob_train,
                                                 gan.is_training: True})



        if(i%10000==0):
            print('discriminator loss:', d_ls)
            print('generator loss:', g_ls)
            print(i)
            n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
            feed_dict = {
                gan.generator_input: n,
                gan.keep_prob: 1,
                gan.is_training: False}
            test_img = sess.run(gan.generator, feed_dict=feed_dict)
            fig = plt.figure(figsize=(8, 8))
            columns = 8
            rows = 8
            for k in range(1, columns * rows + 1):
                img = test_img[k - 1]
                fig.add_subplot(rows, columns, k)
                plt.imshow(np.reshape(img, (28, 28)))
                fig.savefig('results/res_mnist_' + str(i) + '.png')

            saver.save(sess, 'log/mnist_new')
