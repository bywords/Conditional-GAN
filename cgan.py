# -*- encoding: utf-8 -*-
from utils import save_samples, sample_output
from tensorflow.contrib.layers.python.layers import xavier_initializer
from ops import lrelu, fully_connect, batch_normal
import tensorflow as tf
import numpy as np

class CGAN(object):

    # build model
    def __init__(self, data_ob, sample_dir, output_size, learn_rate, batch_size, z_dim, y_dim, log_dir
         , model_path, visua_path, y_min, y_max):

        self.data_ob = data_ob
        self.sample_dir = sample_dir
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.log_dir = log_dir
        self.model_path = model_path
        self.vi_path = visua_path
        self.real_feature_vector = tf.placeholder(tf.float32, [self.batch_size, self.output_size])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim])
        self.y_min = y_min
        self.y_max = y_max

    def build_model(self):

        self.fake_feature_vector = self.gern_net(self.z, self.y)
        G_feature_vector = tf.summary.tensor_summary("G_out", self.fake_feature_vector)
        ##the loss of gerenate network
        D_pro, D_logits = self.dis_net(self.real_feature_vector, self.y, False)
        D_pro_sum = tf.summary.histogram("D_pro", D_pro)

        G_pro, G_logits = self.dis_net(self.fake_feature_vector, self.y, True)
        G_pro_sum = tf.summary.histogram("G_pro", G_pro)

        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(G_pro), logits=G_logits))
        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_pro), logits=D_logits))
        G_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(G_pro), logits=G_logits))

        self.D_loss = D_real_loss + D_fake_loss
        self.G_loss = G_fake_loss

        loss_sum = tf.summary.scalar("D_loss", self.D_loss)
        G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)

        self.merged_summary_op_d = tf.summary.merge([loss_sum, D_pro_sum])
        self.merged_summary_op_g = tf.summary.merge([G_loss_sum, G_pro_sum, G_feature_vector])

        t_vars = tf.trainable_variables()
        self.d_var = [var for var in t_vars if 'dis' in var.name]
        self.g_var = [var for var in t_vars if 'gen' in var.name]

        self.saver = tf.train.Saver()

    def train(self):

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=0.5).minimize(self.D_loss, var_list=self.d_var)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=0.5).minimize(self.G_loss, var_list=self.g_var)
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_writer = tf.summary.FileWriter(self.log_dir, graph=sess.graph)

            step = 0
            while step <= 10000:

                realbatch_array, real_labels = self.data_ob.getNext_batch(step)

                # Get the z
                batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])

                _, summary_str = sess.run([opti_D, self.merged_summary_op_d],
                                          feed_dict={self.real_feature_vector: realbatch_array,
                                                     self.z: batch_z, self.y: real_labels})
                summary_writer.add_summary(summary_str, step)

                _, summary_str = sess.run([opti_G, self.merged_summary_op_g],
                                          feed_dict={self.z: batch_z, self.y: real_labels})
                summary_writer.add_summary(summary_str, step)

                if step % 50 == 0:

                    D_loss = sess.run(self.D_loss,
                                      feed_dict={self.real_feature_vector: realbatch_array,
                                                 self.z: batch_z, self.y: real_labels})
                    fake_loss = sess.run(self.G_loss, feed_dict={self.z: batch_z, self.y: real_labels})
                    print("Step %d: D: loss = %.7f G: loss=%.7f " % (step, D_loss, fake_loss))

                if np.mod(step, 50) == 1 and step != 0:
                    sample_exec_time = sample_output(batch_size=self.batch_size, min=self.y_min, max=self.y_max)
                    sample_feature = sess.run(self.fake_feature_vector, feed_dict={self.z: batch_z,
                                                                                   self.y: sample_exec_time})
                    save_samples(sample_feature, sample_exec_time)
                    self.saver.save(sess, self.model_path)

                step = step + 1

            save_path = self.saver.save(sess, self.model_path)
            print ("Model saved in file: %s" % save_path)

    def test(self):

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            self.saver.restore(sess, self.model_path)
            sample_z = np.random.uniform(1, -1, size=[self.batch_size, self.z_dim])
            sample_exec_time = sample_output(self.batch_size)
            sample_feature = sess.run(self.fake_feature_vector, feed_dict={self.z: sample_z,
                                                                           self.y: sample_exec_time})
            save_samples(sample_feature, sample_exec_time)
            print("Test finish!")

    def gern_net(self, z, y):

        with tf.variable_scope('generator') as scope:

            z = tf.concat([z, y], 1)
            d1 = tf.nn.relu(batch_normal(fully_connect(z, output_size=11, scope="gen_fully1"), scope="gen_bn1"))
            d2 = tf.nn.relu(batch_normal(fully_connect(d1, output_size=11, scope="gen_fully2"), scope="gen_bn2"))
            return fully_connect(d2, output_size=4, scope="gen_fully3", initializer=xavier_initializer())

    def dis_net(self, feature_vector, y, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            # concat
            concat_data = tf.concat([feature_vector, y], 1)
            f1 = tf.nn.relu(batch_normal(fully_connect(concat_data, output_size=10, scope="dis_fully1"), scope="dis_bn1"))
            f1 = tf.concat([f1, y], 1)
            f2 = lrelu(batch_normal(fully_connect(f1, output_size=10, scope="dis_fully1"), scope="dis_bn1"))
            f2 = tf.concat([f2, y], 1)
            out = fully_connect(f2, output_size=1, scope='dis_fully2',  initializer=xavier_initializer())

            return tf.nn.sigmoid(out), out







