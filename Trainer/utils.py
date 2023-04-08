import tensorflow as tf
from matplotlib import pyplot as plt

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
                                tf.keras.layers.GlobalAveragePooling2D(),
                                ])

def save_plots(history, save_folder, model_name):
    loss_path = save_folder + "loss.png"
    acc_path = save_folder + "acc.png"

    for (metric, path) in [('loss', loss_path),('accuracy', acc_path)]:
        plt.plot(history.history[metric])
        plt.plot(history.history[f"val_{metric}"])
        plt.title(model_name)
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(path)
        plt.clf()
