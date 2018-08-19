import model_cifar
from data_utils import *
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from random import sample
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

batch_size = 64
n_noise = 64
num_epoch = 300

image = load_all_images('cifar-10-batches-py')

images_norm = normalized_data(image)

tf.reset_default_graph()

with tf.Session() as sess:
    gan = model_cifar.Gan()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, 'log/cifar_sig_-299')
    for epoch in range(num_epoch):

        images_norm = shuffle(images_norm)

        for i in range(0, len(images_norm) - len(images_norm) % 64, 64):
            # print(len(images_norm) - len(images_norm) % 64)
            train_d = True
            train_g = True
            # print(train_d, train_g)
            keep_prob_train = 0.5  # 0.5

            # n = tf.random_uniform([batch_size, n_noise], dtype=tf.float32)
            n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
            # batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]

            batch = sample(list(images_norm), 64)

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

            if g_ls * 1.35 < d_ls:
                train_g = False
                pass
            if d_ls * 1.35 < g_ls:
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

                # sess.run(gan.optimizer_g, feed_dict={gan.generator_input: n,
                #                                      gan.keep_prob: keep_prob_train,
                #                                      gan.is_training: True})

        print('epoch: ', epoch)
        print('discriminator loss:', d_ls)
        print('generator loss:', g_ls)
        saver.save(sess, 'log/try', global_step=epoch)
        if epoch % 5 == 0:
            n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)
            feed_dict = {
                gan.generator_input: n,
                gan.keep_prob: 1,
                gan.is_training: False}
            test_img = sess.run(gan.generator, feed_dict=feed_dict)
            fig = plt.figure(figsize=(8, 8))
            columns = 8
            rows = 8
            for i in range(1, columns * rows + 1):
                img = test_img[i - 1]
                fig.add_subplot(rows, columns, i)
                plt.imshow(np.reshape(img, (32, 32, 3)))
                fig.savefig('results/res_sig_' + str(epoch) + '.png')
