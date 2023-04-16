import tensorflow as tf
from matplotlib import pyplot as plt
import os
MAG_MAX = 35

BANDS = ["U",
             "F378",
             "F395",
             "F410",
             "F430",
             "G",
             "F515",
             "R",
             "F660",
             "I",
             "F861",
             "Z"]


def eval_string(model_type, model_name, wise, ds_name, total, with_wise, no_wise):
    return f"""Model: {model_type}\nModel Name: {model_name}\nWise: {wise}\nData set name: {ds_name}\nTotal:\n{total}With Wise:\n{with_wise}Without Wise:\n{no_wise}"""

def log_string(model_type, model_name, wise, params, notes):
    return f"""Model: {model_type}\nModel Name: {model_name}\nWise: {wise}\n{params}\nNotes: {notes}\n"""

def vgg16(wise, l2):
    n_channels = 14 if wise else 12
    return tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(32,32,n_channels)),
                                tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                                tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                                tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                                tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                                tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
                                tf.keras.layers.LeakyReLU(),
                                tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
                                ])

def vgg16_decoder(wise,l2):
  n_channels = 14 if wise else 12
  return tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,1,512)), 
    tf.keras.layers.UpSampling2D(size=(2,2)),
    tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.UpSampling2D(size=(2,2)),
    tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(512, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.UpSampling2D(size=(2,2)),
    tf.keras.layers.Conv2DTranspose(256, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(256, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(256, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.UpSampling2D(size=(2,2)),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.UpSampling2D(size=(2,2)),
    tf.keras.layers.Conv2DTranspose(64, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(n_channels, kernel_size=(3,3), padding="same", kernel_regularizer = tf.keras.regularizers.l2(l2)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape([32,32,n_channels])
  ])
   

def save_plots(history, save_folder, model_name):
    loss_path = save_folder + "loss.png"
    acc_path = save_folder + "acc.png"

    vals = [('loss', loss_path),('accuracy', acc_path)] if 'accuracy' in history.history.keys() else [('loss', loss_path)]

    for (metric, s_path) in vals:
        plt.plot(history.history[metric])
        plt.plot(history.history[f"val_{metric}"])
        plt.title(model_name)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if os.path.exists(s_path):
           s_path = s_path[0:-4] + "_f.png"
        plt.savefig(s_path)
        plt.clf()

def pretrain_eval_string(mag_mae, mae):
  return str(dict(zip(BANDS, mag_mae))) + "\n" + "MAE: " + str(mae)

   

class CustomMAE(tf.keras.losses.Loss):
  def __init__(self):
    super().__init__()
    self.mae = tf.keras.losses.MeanAbsoluteError()
  def call(self, y_true, y_pred):
    mvalue = 99/MAG_MAX
    mask = tf.keras.backend.cast(tf.keras.backend.not_equal(y_true, mvalue), tf.keras.backend.floatx())
    return self.mae(y_true*mask, y_pred*mask)