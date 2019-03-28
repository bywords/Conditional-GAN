from cgan import CGAN
from utils import Dataset
import tensorflow as tf
import os

flags = tf.app.flags

flags.DEFINE_string("sample_dir" , "samples_for_test" , "the dir of sample images")
flags.DEFINE_integer("output_size", 4 , "the size of generate image")
flags.DEFINE_float("learn_rate", 0.0002, "the learning rate for gan")
flags.DEFINE_integer("batch_size", 64, "the batch number")
flags.DEFINE_integer("z_dim", 10, "the dimension of noise z")
flags.DEFINE_integer("y_dim", 1, "the dimension of condition y")
flags.DEFINE_string("log_dir", "/tmp/tensorflow_mnist" , "the path of tensorflow's log")
flags.DEFINE_string("model_path", "model/model.ckpt" , "the path of model")
flags.DEFINE_string("visua_path", "visualization" , "the path of visuzation images")
flags.DEFINE_integer("op" , 0, "0: train ; 1:test")
flags.DEFINE_float("y_min", -2.06, "the minimum value for Y (condition)")
flags.DEFINE_float("y_max", 11.44, "the maximum value for Y (condition)")

FLAGS = flags.FLAGS
#
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
if not os.path.exists(FLAGS.model_path):
    os.makedirs(FLAGS.model_path)
if not os.path.exists(FLAGS.visua_path):
    os.makedirs(FLAGS.visua_path)

def main(_):

    data_object = Dataset()

    cg = CGAN(data_ob=data_object, sample_dir=FLAGS.sample_dir, output_size=FLAGS.output_size,
              learn_rate=FLAGS.learn_rate, batch_size=FLAGS.batch_size, z_dim=FLAGS.z_dim, y_dim=FLAGS.y_dim,
              log_dir=FLAGS.log_dir, model_path=FLAGS.model_path, visua_path=FLAGS.visua_path,
              y_min=FLAGS.y_min, y_max=FLAGS.y_max)

    cg.build_model()

    if FLAGS.op == 0:
        cg.train()
    elif FLAGS.op == 1:
        cg.test()
    else:
        print("op should be either 0 or 1.")
        assert(False)


if __name__ == '__main__':
    tf.app.run()
