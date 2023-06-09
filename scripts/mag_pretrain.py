import sys
sys.path.append("..")
from Trainer.data_manager import load_data
from Trainer.sky_classifier import SkyClassifier
import tensorflow as tf

def main():
    print("Loading Data", flush=True)
    ds_name = 'unl_w99'
    data = load_data(ds_name, False)
    X,y = data['images_train'], data['tabular_train'][:,0:12]
    X_val,y_val = data['images_val'], data['tabular_val'][:,0:12]
    X_test,y_test = data['images_test'], data['tabular_test'][:,0:12]
    print("Loaded Data", flush=True)

    lr = float(sys.argv[1])
    l2 = float(sys.argv[2])
    dpout = float(sys.argv[3])
    model_name = f"vgg16mod_pretrain_mags_lr_{lr}_l2_{l2}_dt_{dpout}"

    print(model_name, flush=True)
    model = SkyClassifier("vgg16", model_name,
                           False, pretext_output='magnitudes')
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.build_model(l2=l2, dropout=dpout, opt=opt)
    model.pretrain(X, y, X_val=X_val, y_val=y_val, notes = f"lr: {lr}, l2: {l2}, dropout: {dpout}", epochs=100)
    model.load_model()
    model.eval_pretrain(X_val, y_val, "unl_w99_val")
    model.eval_pretrain(X_test, y_test, "unl_w99_test")

if __name__ == "__main__":
    main()