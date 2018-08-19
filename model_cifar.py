import tensorflow as tf
import tensorflow.contrib as tfc


def lrelu(x):
    return tf.nn.leaky_relu(x)


def binary_cross_entropy(x, z):
    eps = 1e-12
    return -(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps))


class Gan:
    def __init__(self):
        self.discriminator_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='d_input')
        self.generator_input = tf.placeholder(dtype=tf.float32, shape=[None, 64])
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        self.generator = self.build_generator(self.generator_input, keep_prob=0.5)
        self.discriminator_real = self.build_discriminator(self.discriminator_input, self.keep_prob)
        self.discriminator_fake = self.build_discriminator(self.generator, self.keep_prob, reuse=True)
        self.build_loss()

    def build_generator(self, inp, keep_prob):
        activation = lrelu
        is_training = self.is_training
        momentum = 0.99
        with tf.variable_scope('generator') as scope:
            x = inp
            x = tf.layers.dense(x, units=16, activation=activation)
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.reshape(x, shape=[-1, 4, 4, 1])
            x = tf.image.resize_images(x, size=[8, 8])
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same')
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same',
                                           activation=activation)
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same',
                                           activation=activation)
            x = tf.layers.dropout(x, keep_prob)
            # x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=1, padding='same',
                                           activation=tf.nn.sigmoid)
            print(x.shape)
        return x

    def build_discriminator(self, img_in, keep_prob, reuse=None):
        momentum = 0.99
        is_training = self.is_training
        activation = lrelu
        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.reshape(img_in, shape=[-1, 32, 32, 3])
            x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
            # x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.layers.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=1, padding='same', activation=activation)
            # x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.layers.dropout(x, keep_prob)
            x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
            # x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=128, activation=activation)
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x

    def build_loss(self):
        vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
        vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

        d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
        g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

        self.loss_d_real = binary_cross_entropy(tf.ones_like(self.discriminator_real),
                                                self.discriminator_real)
        print(tf.shape(tf.ones_like(self.discriminator_real)))
        self.loss_d_fake = binary_cross_entropy(tf.zeros_like(self.discriminator_fake),
                                                self.discriminator_fake)
        self.loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(self.discriminator_fake),
                                                          self.discriminator_fake))
        self.loss_d = tf.reduce_mean(0.5 * (self.loss_d_fake + self.loss_d_real))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.optimizer_d = tf.train.AdamOptimizer(0.0005).minimize(self.loss_d + d_reg,
                                                                      var_list=vars_d)
            self.optimizer_g = tf.train.AdamOptimizer(0.0005).minimize(self.loss_g + g_reg,
                                                                       var_list=vars_g)
