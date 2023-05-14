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

    with tf.device("CPU"):
        X = tf.data.Dataset.from_tensor_slices(X[30000:60000]).batch(1)
        #X_val = tf.data.Dataset.from_tensor_slices(X_val).batch(1)
        #X_test = tf.data.Dataset.from_tensor_slices(X_test).batch(1)
        
    
    print("model ready", flush=True)
    clean_X = model.predict(X)
    print("train done", flush=True)
    #clean_X_val = model.predict(X_val)
    print("val done", flush=True)
    #clean_X_test = model.predict(X_test)
    print("test done", flush=True)

    print(clean_X.shape)
    np.save('./../Data/ready/clf/temp2.npy', clean_X)
    #np.save('./../Data/ready/clf/clf_90_5_5_clean-images_train.npy', clean_X)
    print("train saved", flush=True)
    #np.save('./../Data/ready/clf/clf_90_5_5_clean-images_test.npy', clean_X_test)
    print("test saved", flush=True)
    #np.save('./../Data/ready/clf/clf_90_5_5_clean-images_val.npy', clean_X_val)
    print("val saved", flush=True)

if __name__ == "__main__":
    main()