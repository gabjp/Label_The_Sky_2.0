import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report
from Trainer.utils import eval_string, log_string, vgg16, save_plots, CustomMAE, MAG_MAX, pretrain_eval_string, vgg16_decoder, save_sample
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight



MODELS = ['vgg16', 'RF']
CLASS_NAMES = ['QSO', 'STAR', 'GALAXY']
PRETEXT_OUTPUT_TYPES = ['magnitudes', 'images', None]
SEED = 2
OUT_DIR = "../outs/" # This assumes that the program will be run from scripts dir

class SkyClassifier:
    def __init__(self, model, model_name, wise, pretext_output=None, save=True):
        if model not in MODELS: raise ValueError(f"{model} not in {MODELS}")
        self.model_type = model

        if pretext_output not in PRETEXT_OUTPUT_TYPES: 
            raise ValueError(f"{pretext_output} not in {PRETEXT_OUTPUT_TYPES}")
        self.pretext_output = pretext_output

        self.wise = wise
        self.save = save
        self.model_name = model_name
        self.model_folder = OUT_DIR+self.model_name + '/'

        if self.save and not os.path.exists(OUT_DIR+self.model_name):
            os.mkdir(OUT_DIR+self.model_name)

        if self.model_type == "vgg16":
            self.callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.model_folder + self.model_name + ".h5"),
                    monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min')
            ] if self.save else None

    def build_model(self, to_finetune = False, **kwargs):
        if self.model_type == "RF":
            self.model = RandomForestClassifier(random_state = SEED, **kwargs)
            

        elif self.model_type == "vgg16":
            print(f"Checking GPU: {tf.config.list_physical_devices('GPU')}")
            tf.random.set_seed(SEED) # Set global seed
            
            l2 = kwargs['l2'] if 'l2' in kwargs.keys() else 0
            dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0

            if self.pretext_output == None:
                self.model = vgg16(self.wise, l2)

                if to_finetune:
                    for layer in self.model.layers:
                        layer.trainable = False

                self.model.add(tf.keras.layers.GlobalAveragePooling2D()),
                self.model.add(tf.keras.layers.Dense(1024, kernel_regularizer = tf.keras.regularizers.l2(l2)))
                self.model.add(tf.keras.layers.LeakyReLU())
                self.model.add(tf.keras.layers.Dropout(dropout))
                self.model.add(tf.keras.layers.Dense(3, activation='softmax'))
                loss = "categorical_crossentropy"
                metrics = ["accuracy"]

            elif self.pretext_output == 'magnitudes':
                self.model = vgg16(self.wise, l2)

                self.model.add(tf.keras.layers.GlobalAveragePooling2D()), 
                self.model.add(tf.keras.layers.Dense(1024, kernel_regularizer = tf.keras.regularizers.l2(l2)))
                self.model.add(tf.keras.layers.LeakyReLU())
                self.model.add(tf.keras.layers.Dropout(dropout))
                if self.wise :
                    self.model.add(tf.keras.layers.Dense(14))
                else:
                    self.model.add(tf.keras.layers.Dense(12))
                
                loss = CustomMAE()
                metrics = None

            elif self.pretext_output == 'images':
                self.encoder = vgg16(self.wise, l2)
                self.decoder = vgg16_decoder(self.wise, l2)
                self.model = tf.keras.Sequential([self.encoder, self.decoder])

                input_shape = (None,32,32,12) if not self.wise else (None,32,32,14)

                self.model.build(input_shape=input_shape)
                loss = "mae"
                metrics = None

            
            if 'opt' not in kwargs.keys(): raise ValueError("missing opt paramenter (with learning rate)") 
            self.model.compile(metrics = metrics, loss=loss, optimizer = kwargs["opt"]) # expects optimizer (with learning rate)

    def pretrain(self, X, y, X_val, y_val, batch_size=32, epochs=100, notes=""):
        self.model.summary()
        if self.pretext_output == 'magnitudes':

            with tf.device("CPU"):
                train = tf.data.Dataset.from_tensor_slices((X, y/MAG_MAX)).batch(batch_size)
                validate = tf.data.Dataset.from_tensor_slices((X_val, y_val/MAG_MAX)).batch(batch_size)
        
        elif self.pretext_output == 'images':

            with tf.device("CPU"):
                train = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
                validate = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        history = self.model.fit(
            train,
            validation_data = validate,
            batch_size = batch_size,
            epochs = epochs,
            callbacks =  self.callbacks,
            verbose=2
        )

        self.load_model()
        self.decoder.save_weights("decoder.h5")

        if self.save:
            with open(self.model_folder + 'log', 'w') as log:
                log.write(log_string(self.model_type, self.model_name, self.wise, self.model.to_json() , notes))
            save_plots(history, self.model_folder, self.model_name)

            
            

    def eval_pretrain(self, X, y, ds_name, save_img= False):
        
        if self.pretext_output == "magnitudes":

            y_hat = self.model.predict(X) * MAG_MAX
            err = np.abs(y-y_hat)
            mag_mae = err.mean(axis=0)
            mae = mag_mae.mean()

            output = pretrain_eval_string(mag_mae, mae)

        elif self.pretext_output == "images":
            y_hat = self.model.predict(X) 
            err = np.abs(y-y_hat)
            mag_mae = err.mean(axis=(0,1,2)) # (n,32,32,12)
            mae = mag_mae.mean()

            output = pretrain_eval_string(mag_mae, mae)

            if save_img:
                for i in range(5):
                    save_sample(self.model_folder, str(i), y[i], y_hat[i])

        if self.save:
            with open(self.model_folder + ds_name + '.results', 'w') as results:
                results.write(output)
                
        print(output)


    def train(self, X, y, X_val=None, y_val=None,epochs=100, batch_size=32, notes=None):
        if self.model_type == "RF":
            self.model.fit(X, y=y)

            if self.save: 
                joblib.dump(self.model, self.model_folder + self.model_name + '.sav')
                with open(self.model_folder + 'log', 'w') as log:
                    log.write(log_string(self.model_type, self.model_name, self.wise, self.model.get_params(), notes))

        elif self.model_type == "vgg16" and self.pretext_output == None:
            if type(X_val) == type(None) or type(y_val) == type(None):
                raise ValueError("Add validation data")
            
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
            class_weights = {0:class_weights[0],1:class_weights[1],2:class_weights[2]}
            self.model.summary()

            with tf.device("CPU"):
                train = tf.data.Dataset.from_tensor_slices((X, tf.keras.utils.to_categorical(y))).batch(batch_size)
                validate = tf.data.Dataset.from_tensor_slices((X_val, tf.keras.utils.to_categorical(y_val))).batch(batch_size)

            history = self.model.fit(
                train,
                validation_data=validate,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=self.callbacks,
                class_weight=class_weights,
                verbose=2
            )
            if self.save:
                with open(self.model_folder + 'log', 'w') as log:
                    log.write(log_string(self.model_type, self.model_name, self.wise, self.model.to_json() , notes))
                save_plots(history, self.model_folder, self.model_name)

        else:
            raise ValueError("You need to pretrain/finetune a model with pretext_output != none")

    def finetune(self, X, y, X_val, y_val,f_opt ,batch_size=32, w_epochs=10, f_epochs= 100, notes=None):
        self.train(X,y,X_val,y_val, batch_size=batch_size, epochs=w_epochs, notes=notes)
        for layer in self.model.layers:
            layer.trainable = True
        self.model.compile(metrics = ["accuracy"], loss="categorical_crossentropy", optimizer = f_opt)
        self.train(X,y,X_val,y_val, batch_size=batch_size, epochs=f_epochs, notes=notes)

    def load_model(self, weights_path = None, finetune=False):

        weights_path = self.model_folder + self.model_name if weights_path == None else weights_path

        if self.model_type == "RF":
            self.model = joblib.load(weights_path + '.sav')
        
        elif self.model_type == "vgg16":
            if finetune:
                self.model.load_weights(weights_path + '.h5', by_name=True, skip_mismatch=True)
            else:
                self.model.load_weights(weights_path + '.h5')

    def eval(self, X, y, ds_name, wise_flags = None):
        if self.model_type == "RF":
            pred = self.model.predict(X)

        elif self.model_type == "vgg16":
            pred = np.argmax(self.model.predict(X), axis=1)

        total = classification_report(y, pred, digits = 6, target_names = CLASS_NAMES)
        with_wise = classification_report(y[wise_flags], pred[wise_flags], digits = 6, target_names = CLASS_NAMES) if type(wise_flags) != type(None) else None
        no_wise = classification_report(y[np.invert(wise_flags)], pred[np.invert(wise_flags)], digits = 6, target_names = CLASS_NAMES) if type(wise_flags) != type(None) else None

        output = eval_string(self.model_type, self.model_name, self.wise, ds_name, total, with_wise, no_wise)

        if self.save:
            with open(self.model_folder + ds_name + '.results', 'w') as results:
                results.write(output)
                
        print(output)
        
    def predict(self):
        pass

    def predict_proba(self):
        pass