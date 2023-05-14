import sys
sys.path.append("..")
from Trainer.data_manager import load_data
from Trainer.sky_classifier import SkyClassifier
import tensorflow as tf

def main():
    print("Loading Data", flush=True)
    ds_name = 'clf_90_5_5'
    data = load_data(ds_name, True, True)
    X,y = data['images_train'], data['class_train']
    X_val,y_val = data['images_val'], data['class_val']
    X_test,y_test = data['images_test'], data['class_test']
    wise_val, wise_test = data['wiseflags_val'], data['wiseflags_test'] 
    print("Loaded Data", flush=True)

    lr = float(sys.argv[1])
    l2 = float(sys.argv[2])
    dpout = float(sys.argv[3])
    model_name = f"vgg16clean_from_scratch_lr_{lr}_l2_{l2}_dt_{dpout}"

    print(model_name, flush=True)
    model = SkyClassifier(f"vgg16", model_name, False)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.build_model(l2=l2, dropout=dpout, opt=opt)
    model.train(X, y, X_val=X_val, y_val=y_val, notes = f"lr: {lr}, l2: {l2}, dropout: {dpout}", epochs=100)
    model.load_model()
    model.eval(X_val, y_val, "clf_90_5_5_val", wise_flags=wise_val)
    model.eval(X_test, y_test, "clf_90_5_5_test", wise_flags=wise_test)

if __name__ == "__main__":
    main()
    
