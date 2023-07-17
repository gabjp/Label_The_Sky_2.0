import sys
sys.path.append("..")
from Trainer.data_manager import load_data
from Trainer.sky_classifier import SkyClassifier
import numpy as np
import tensorflow as tf

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

    np.save("../preds/true_nwval.npy", y_val[np.invert(wise_val)])
    np.save("../preds/true_nwtest.npy", y_test[np.invert(wise_test)])

    X_val,y_val = data['tabular_val'], data['class_val']
    X_test,y_test = data['tabular_test'], data['class_test']

    for i in range(1,4):
        unified_rf = SkyClassifier('RF', f"Unified_RF_run{i}", True, save=False)
        unified_rf.build_model(n_estimators=100, bootstrap=False)
        unified_rf.load_model(weights_path=f"../outs/RFs/Unified_RF_run{i}/Unified_RF_run{i}")

        unified_rf.eval(X_val, y_val,"clf_90_5_5_val", wise_flags=wise_val)

        pred_val = unified_rf.predict_proba(X_val[np.invert(wise_val)])
        pred_test = unified_rf.predict_proba(X_test[np.invert(wise_test)])
        np.save("../preds/URF_{i}_nwval.npy", pred_val)
        np.save("../preds/URF_{i}_nwtest.npy", pred_test)

    X_val,y_val = data['images_val'], data['class_val']
    X_test,y_test = data['images_test'], data['class_test']

    for i in range (1,4):
        model_name = f"vgg16mod_finetune_mags_wlr_0.0001_flr_1e-05_l2_0.0007_dt_0.3_epochs_30_nowiseval_1_run{i}"
        vgg = SkyClassifier("vgg16", model_name, False, save=False)
        opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
        vgg.build_model(to_finetune=False ,l2=0.0007, dropout=0.3, opt=opt)
        vgg.load_model()

        vgg.eval(X_val, y_val,"clf_90_5_5_val", wise_flags=wise_val)

        pred_val = vgg.predict(X_val[np.invert(wise_val)])
        pred_test = vgg.predict(X_test[np.invert(wise_test)])
        np.save("../preds/VGG_{i}_nwval.npy", pred_val)
        np.save("../preds/VGG_{i}_nwtest.npy", pred_test)


if __name__ == "__main__":
    main()