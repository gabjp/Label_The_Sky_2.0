import sys
sys.path.append("..")
from Trainer.data_manager import load_data
from Trainer.sky_classifier import SkyClassifier
import tensorflow as tf
import numpy as np

def main():
    print("Loading Data", flush=True)
    ds_name = 'clf_90_5_5'
    data = load_data(ds_name, True)
    X,y = data['images_train'], data['class_train']
    X_val,y_val = data['images_val'], data['class_val']
    X_test,y_test = data['images_test'], data['class_test']
    wise_val, wise_test = data['wiseflags_val'], data['wiseflags_test']
    wise_train = data['wiseflags_train'] 
    print("Loaded Data", flush=True)

    weights_path = "../outs/vgg16mod_pretrain_mags_lr_1e-05_l2_0.0_dt_0.0_run3/vgg16mod_pretrain_mags_lr_1e-05_l2_0.0_dt_0.0_run3"
    w_lr = float(sys.argv[1])
    f_lr = float(sys.argv[2])
    l2 = float(sys.argv[3])
    dpout = float(sys.argv[4])
    f_epochs = int(sys.argv[5])
    nowiseval = int(sys.argv[6])
    run = int(sys.argv[7])

    if nowiseval:
        X_val = X_val[np.invert(wise_val)]
        y_val = y_val[np.invert(wise_val)]
        X = X[np.invert(wise_train)]
        y = y[np.invert(wise_train)]


    model_name = f"vgg16mod_finetune_mags_wlr_{w_lr}_flr_{f_lr}_l2_{l2}_dt_{dpout}_epochs_{f_epochs}_nowiseval_{nowiseval}_run{run}"

    print(model_name, flush=True)
    model = SkyClassifier("vgg16", model_name, False)

    opt = tf.keras.optimizers.Adam(learning_rate=w_lr)
    model.build_model(to_finetune=True ,l2=l2, dropout=dpout, opt=opt)
    model.load_model(weights_path= weights_path, finetune=True)
    opt = tf.keras.optimizers.Adam(learning_rate=f_lr)
    model.finetune(X,y, X_val, y_val, opt, notes= f"w_lr: {w_lr}, f_lr: {f_lr}", f_epochs=f_epochs)

    X_val,y_val = data['images_val'], data['class_val']
    wise_val, wise_test = data['wiseflags_val'], data['wiseflags_test']

    model.load_model()
    model.eval(X_val, y_val, "clf_90_5_5_val", wise_flags=wise_val)
    model.eval(X_test, y_test, "clf_90_5_5_test", wise_flags=wise_test)

if __name__ == "__main__":
    main()