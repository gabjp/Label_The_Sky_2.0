from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from Trainer.sky_classifier import OUT_DIR
from sklearn.preprocessing import StandardScaler


class MetaTrainer:
    def __init__(self, model_name, save=True):

        self.save = save
        self.model_name = model_name
        self.model_folder = OUT_DIR + self.model_name + '/'

        if self.save and not os.path.exists(OUT_DIR+self.model_name):
            os.mkdir(OUT_DIR+self.model_name)

        self.model = keras.models.Sequential()

        self.model.add(keras.layers.Input(shape=(6,)))
        self.model.add(keras.layers.Dense(300, activation = "relu"))
        self.model.add(keras.layers.Dense(100, activation = "relu"))
        self.model.add(keras.layers.Dense(3, activation = "softmax"))
        self.model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.Adam(lr=1e-3), metrics = ["accuracy"])
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.ss = StandardScaler()
        self.ss.fit(X_train)

        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = tf.keras.utils.to_categorical(y_val)

        t_X_train = self.ss.transform(X_train)
        t_X_val = self.ss.transform(X_val)

        self.model.fit(t_X_train, y_train,validation_data = (t_X_val, y_val), batch_size = 32, verbose = 2, epochs = 30, 
            class_weight=compute_class_weight(class_weight='balanced', classes=[0,1,2], y = np.argmax(y_train, axis=1)),
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                        filepath=self.model_folder + self.model_name + ".h5",
                        save_weights_only=True,
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True)])

        self.model.load_weights(self.model_folder + self.model_name + ".h5")

    def predict_proba(self, X):
        t_X = self.ss.transform(X)
        return self.model.predict(t_X)