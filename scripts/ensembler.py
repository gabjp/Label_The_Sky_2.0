import numpy as np
import sys
sys.path.append("..")
from Trainer.data_manager import load_data, CLF_READY_FOLDER
from sklearn.model_selection import StratifiedKFold
from Trainer.sky_classifier import SkyClassifier
import tensorflow as tf


PRED_DIR = "../Data/ready/meta/"

def generate_data():

    print("Loading Data", flush=True)
    ds_name = 'clf_90_5_5'
    data = load_data(ds_name, True, False)
    print("Loaded Data", flush=True)
    print(data.keys())


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    split = list(skf.split(data["tabular_train"], data["class_train"]))

    RF_pred = np.array([]).reshape(0,3)
    target = np.array([])
    wise_flags = np.array([])

    for i, (train_index, gen_index) in enumerate(split):
        print(f"Training RF_{i}", flush=True)
        rf = SkyClassifier("RF", f"RF_{i}", True, save=False)
        rf.build_model(n_estimators=100, bootstrap=False)
        rf.train(data["tabular_train"][train_index], data["class_train"][train_index])
        probs = rf.predict_proba(data["tabular_train"][gen_index])

        RF_pred = np.concatenate((RF_pred, probs), axis = 0)
        target = np.concatenate((target, data["class_train"][gen_index]), axis=0)
        wise_flags = np.concatenate((wise_flags, data["wiseflags_train"][gen_index]), axis=0)
        
    weights_path = "../outs/vgg16mod_pretrain_mags_lr_1e-05_l2_0.0_dt_0.0/vgg16mod_pretrain_mags_lr_1e-05_l2_0.0_dt_0.0"

    VGG_pred = np.array([]).reshape(0,3)

    for i, (train_index, gen_index) in enumerate(split):
        print(f"Training VGG_{i}", flush=True)
        vgg = SkyClassifier("vgg16", f"vgg16_{i}", False)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        vgg.build_model(to_finetune=True ,l2=0, dropout=0.3, opt=opt)
        vgg.load_model(weights_path= weights_path, finetune=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        vgg.finetune(data["images_train"][train_index], data["class_train"][train_index],
                      data["images_val"], data["class_val"], opt, f_epochs=100)
        vgg.load_model()

        probs = vgg.predict(data["images_train"][gen_index])
        VGG_pred = np.concatenate((VGG_pred, probs), axis = 0)

    meta_features = np.concatenate((RF_pred, VGG_pred), axis = 1)

    print("Saving meta-model trainig set", flush=True)
    np.save(PRED_DIR + "meta_features.npy",meta_features)
    np.save(PRED_DIR +"meta_target.npy",target)
    np.save(PRED_DIR +"meta_wise.npy",wise_flags)



    return 

def train():
    pass

def eval():
    pass

def main():
    if 'g' in sys.argv[1]:
        generate_data()
    pass

if __name__ == "__main__":
    main()