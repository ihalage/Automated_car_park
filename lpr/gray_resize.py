import glob
import cv2

def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)
        img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(img, (256,32))
        save_fname = fname.split("/")[1]

        save_fname1= save_fname.split("_")[1]
        num = save_fname.split("_")[0]
        #save_fname2 = save_fname1.split(".")[0]
        #print(save_fname1)
        #png = save_fname1.split(".")[1]

        save = "tes2" + "/" + num + "_" + save_fname1 + "_" + "1" + "." + "jpg"
        #print fname
        cv2.imwrite(save, resized_img)

read_data("crop/*.jpg")
