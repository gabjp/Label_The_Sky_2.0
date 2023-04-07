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

    model = SkyClassifier("vgg16", "vgg16_from_scratch", False)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.build_model(l2=0, dropout=0.2, opt=opt)
    model.train(X, y, X_val=X_val, y_val=y_val, notes = f"lr: 1e-3, l2: 0.0007, dropout: 0.5", epochs=10)
    model.eval(X_val, y_val, "clf_90_5_5_val", wise_flags=wise_val)
    model.eval(X_test, y_test, "clf_90_5_5_test", wise_flags=wise_test)

if __name__ == "__main__":
    main()
    
