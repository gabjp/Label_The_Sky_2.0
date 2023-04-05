import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report
from utils import eval_string, log_string


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

        if self.save:
            os.mkdir(OUT_DIR+self.model_name)

    def build_model(self, **kwargs):
        if self.model_type == "RF":
            self.model = RandomForestClassifier(random_state = SEED, **kwargs)
            

        elif self.model_type == "vgg16":
            pass

    def finetune(self):
        pass

    def pretrain(self):
        pass

    def train(self, X, y, notes):
        if self.model_type == "RF":
            self.model.fit(X, y=y)

            if self.save: 
                joblib.dump(self.model, self.model_folder + self.model_name + '.sav')
                with open(self.model_folder + self.model_name + '.log', 'w') as log:
                    log.write(log_string(self.model_type, self.model_name, self.wise, self.model.get_params(), notes))

        elif self.model_type == "vgg16":
            pass


    def load_model(self):
        if self.model_type == "RF":
            self.model = joblib.load(self.model_folder + self.model_name + '.sav')
        
        elif self.model_type == "vgg16":
            pass

    def eval(self, X, y, ds_name, wise_flags = None):
        if self.model_type == "RF":
            with open(self.model_folder + self.model_name + '_' + ds_name + '.results', 'w') as results:
                    pred = self.model.predict(X)
                    total = classification_report(y, pred, digits = 6, target_names = CLASS_NAMES)
                    with_wise = classification_report(y[wise_flags], pred[wise_flags], digits = 6, target_names = CLASS_NAMES) if type(wise_flags) != type(None) else None
                    no_wise = classification_report(y[np.invert(wise_flags)], pred[np.invert(wise_flags)], digits = 6, target_names = CLASS_NAMES) if type(wise_flags) != type(None) else None

                    output = eval_string(self.model_type, self.model_name, self.wise, ds_name, total, with_wise, no_wise)
                    
                    if self.save:
                        results.write(output)
                    else:
                        print(output)
            


        elif self.model_type == "vgg16":
            pass
        
    def predict(self):
        pass

    def predict_proba(self):
        pass