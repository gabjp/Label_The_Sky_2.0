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
    X,y = data['images_train'], data['images_train']
    X_val,y_val = data['images_val'], data['images_val']
    X_test,y_test = data['images_test'], data['images_test']
    print("Loaded Data", flush=True)

    lr = 0.0001
    l2 = 0.0


    model_name = f"vgg16_4x4_pretrain_img_lr_{lr}_l2_{l2}"

    print(model_name, flush=True)
    model = SkyClassifier("vgg16",model_name,
                           False, pretext_output='images')
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.build_model(l2=l2, opt=opt)

    model.load_model()
    
    clean_X = model.predict(X)
    clean_X_val = model.predict(X_val)
    clean_X_test = model.predict(X_test)

    np.save('./../Data/ready/clf/clf_90_5_5_clean-images_train.npy', clean_X)
    np.save('./../Data/ready/clf/clf_90_5_5_clean-images_test.npy', clean_X_test)
    np.save('./../Data/ready/clf/clf_90_5_5_clean-images_val.npy', clean_X_val)

if __name__ == "__main__":
    main()