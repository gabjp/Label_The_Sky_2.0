import sys
sys.path.append("..")
from Trainer.data_manager import load_data
from Trainer.sky_classifier import SkyClassifier
import tensorflow as tf

def main():
    print("Loading Data", flush=True)
    ds_name = 'clf_90_5_5'
    data = load_data(ds_name, True)
    X,y = data['images_train'], data['class_train']
    X_val,y_val = data['images_val'], data['class_val']
    X_test,y_test = data['images_test'], data['class_test']
    wise_val, wise_test = data['wiseflags_val'], data['wiseflags_test'] 
    print("Loaded Data", flush=True)

    weights_path = "../outs/vgg16_pretrain_mags_lr:0.0001_l2:0.0_dropout:0.5/vgg16_pretrain_mags_lr:0.0001_l2:0.0_dropout:0.5"
    w_lr = float(sys.argv[1])
    f_lr = float(sys.argv[2])
    l2 = float(sys.argv[3])
    dpout = float(sys.argv[4])

    print(f"vgg16_finetune_l2_{l2}_dt_{dpout}", flush=True)
    model = SkyClassifier(f"vgg16", f"vgg16_finetune_l2_{l2}_dropout_{dpout}", False)

    opt = tf.keras.optimizers.Adam(learning_rate=w_lr)
    model.build_model(to_finetune=True ,l2=l2, dropout=dpout, opt=opt)
    model.load_model(weights_path= weights_path, finetune=True)
    opt = tf.keras.optimizers.Adam(learning_rate=f_lr)
    model.finetune(X,y, X_val, y_val, opt, notes= f"w_lr: {w_lr}, f_lr: {f_lr}", f_epochs=300)

    model.load_model()
    model.eval(X_val, y_val, "clf_90_5_5_val", wise_flags=wise_val)
    model.eval(X_test, y_test, "clf_90_5_5_test", wise_flags=wise_test)

if __name__ == "__main__":
    main()