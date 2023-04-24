import sys
sys.path.append("..")
from Trainer.data_manager import load_data
from Trainer.sky_classifier import SkyClassifier
import tensorflow as tf

def main():
    print("Loading Data", flush=True)
    ds_name = 'unl_w99'
    data = load_data(ds_name, False)
    X,y = data['images_train'], data['images_train']
    X_val,y_val = data['images_val'], data['images_val']
    X_test,y_test = data['images_test'], data['images_test']
    print("Loaded Data", flush=True)

    lr = float(sys.argv[1])
    l2 = float(sys.argv[2])
    model_name = f"vgg16_pretrain_img_lr_{lr}_l2_{l2}"

    print(model_name, flush=True)
    model = SkyClassifier("vgg16",model_name,
                           False, pretext_output='images')
    
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.build_model(l2=l2, opt=opt)
    model.pretrain(X, y, X_val=X_val, y_val=y_val, notes = f"lr: {lr}, l2: {l2}", epochs=50)
    model.load_model()
    model.eval_pretrain(X_val, y_val, "unl_w99_val", save_img=True)
    model.eval_pretrain(X_test, y_test, "unl_w99_test")

if __name__ == "__main__":
    main()