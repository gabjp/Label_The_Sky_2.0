from astropy.io import fits
import pandas as pd
import numpy as np
from cv2 import resize, INTER_CUBIC
import sys
from tqdm import tqdm
import multiprocessing

# 0 -> QSO; 1 -> STAR; 2 -> GAL

RA = 'RA'
DEC = 'DEC'
CLF_FITS_FOLDER = '../Data/raw/clf/'
UNL_FITS_FOLDER = '../Data/raw/unl/'
CLF_READY_FOLDER = '../Data/ready/clf/'
UNL_READY_FOLDER = '../Data/ready/unl/'
CLF_TABLE_FOLDER = '../Data/tables/'
UNL_TABLE_FOLDER = '../Data/tables/'
ZP_TABLE_PATH = '../Data/tables/iDR4_zero-points.csv'

BANDS = ["U",
             "F378",
             "F395",
             "F410",
             "F430",
             "G",
             "F515",
             "R",
             "F660",
             "I",
             "F861",
             "Z"]

BAND_TO_ZP = {"U":"ZP_u",
            "F378":"ZP_J0378",
            "F395":"ZP_J0395",
            "F410":"ZP_J0410",
            "F430":"ZP_J0430",
            "G":"ZP_g",
            "F515":"ZP_J0515",
            "R":"ZP_r",
            "F660":"ZP_J0660",
            "I":"ZP_i",
            "F861":"ZP_J0861",
            "Z":"ZP_z"
}

SPLUS_MAGS = ['u_iso', 
              'J0378_iso', 
              'J0395_iso', 
              'J0410_iso', 
              'J0430_iso', 
              'g_iso', 
              'J0515_iso', 
              'r_iso', 
              'J0660_iso', 
              'i_iso', 
              'J0861_iso', 
              'z_iso']

SPLUS_MORPH = ['A','B','KRON_RADIUS', 'FWHM_n']

WISE_MAGS = ['w1mpro', 'w2mpro']

def calibrate(x, id, band, zps):
    ps = 0.55
    zp = float(zps[zps["Field"]==id[7:20]][BAND_TO_ZP[band]])
    return (10**(5-0.4*zp)/(ps*ps))*x

def gather_bands(index_id, fits_folder, all_images, zps):
        return 
        #Computar imagem
        print("oi")
        index, id = index_id 
        all_bands = []

        for band in BANDS:
            fits_path = fits_folder + band + f"/{id}.fits"
            img = fits.open(fits_path)[1].data
            img = calibrate(img, id, band, zps)
            img = resize(img, dsize=(32, 32), interpolation=INTER_CUBIC)
            all_bands.append(img)
            
        final_all_bands = np.transpose(np.array(all_bands), (1,2,0))
        all_images[index,:] = final_all_bands

class Pseudo_lambda(object):
    def __init__(self, fits_folder, all_images, zps):
        self.fits_folder = fits_folder
        self.all_images = all_images
        self.zps = zps
    def __call__(self, index_id):
        gather_bands(index_id, self.fits_folder, self.all_images, self.zps)

def main():
    if len(sys.argv) != 3: print(f"Usage: {sys.argv[0]} <clf/unl> <csv name>")

    is_clf = True if sys.argv[1] == 'clf' else False
    csv_path = CLF_TABLE_FOLDER + sys.argv[2] + ".csv" if is_clf else UNL_TABLE_FOLDER + sys.argv[2] + ".csv"
    fits_folder = CLF_FITS_FOLDER if is_clf else UNL_FITS_FOLDER 
    ready_folder = CLF_READY_FOLDER if is_clf else UNL_READY_FOLDER 
    csv = pd.read_csv(csv_path)
    zps = pd.read_csv(ZP_TABLE_PATH)

    for split in ['train', 'val', 'test']:
        temp_csv = csv[csv.split==split]
        all_images = np.zeros((len(temp_csv.index),) + (32, 32, 12))

        print("Processing fits files")

        index_id = list(enumerate(temp_csv.ID))

        #multiprocessing.cpu_count()
        with multiprocessing.Pool(2) as pool:
            with tqdm(total=len(index_id)) as pbar:
                for _ in pool.imap_unordered(Pseudo_lambda(fits_folder, all_images, zps), index_id):
                    pbar.update(1)


        np.save(ready_folder + f"{sys.argv[2]}_images_{split}.npy", all_images)
        print(f"File Saved: {ready_folder}{sys.argv[2]}_images_{split}.npy")

        columns = SPLUS_MAGS + SPLUS_MORPH + WISE_MAGS if is_clf else SPLUS_MAGS + SPLUS_MORPH 

        np.save(ready_folder + f"{sys.argv[2]}_tabular_{split}.npy", 
                temp_csv[columns].to_numpy())
        print(f"File Saved: {ready_folder}{sys.argv[2]}_tabular_{split}.npy")

        if is_clf:
            np.save(ready_folder + f"{sys.argv[2]}_class_{split}.npy", 
                    temp_csv['target'].to_numpy())
            print(f"File Saved: {ready_folder}{sys.argv[2]}_class_{split}.npy")


    return 

if __name__=="__main__":
    main()