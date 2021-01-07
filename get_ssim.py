from skimage.io import imread
from skimage.transform import resize
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import numpy as np

import os


def get_ssim(a, b):
    a = imread(a, as_gray=True)
    b = imread(b, as_gray=True)
    a_resized = resize(a, (b.shape[0], b.shape[1]), anti_aliasing=True)

    return psnr(a_resized, b), ssim(a_resized, b)


datasets = "NYU_0.84"

test_on = "NYU"
trained_on = "NYU_0.84"

f = open("./result_values/" + test_on + "_" + trained_on + ".txt", "w")
path_to_folder = './result_images/test_' + test_on + '_' + trained_on + '/'

psnr_all = []
ssim_all = []
for i in os.listdir(path_to_folder):
    ms, ssi, st = None, None, ""
    if i.endswith('.jpg'):
        mid = i.split('_')[1]
        ms, ssi = get_ssim(path_to_folder + i,
                           "./datasets/" + datasets + "/testB/" + mid + ".jpg")
        st = i + " PSNR: " + str(ms) + " SSIM: " + str(ssi) + "\n\n"
    elif i.endswith('.png'):
        # mid = i.split('_')[1]
        mid = i.split('.')[0].split('_')[1]
        ms, ssi = get_ssim(path_to_folder + i, "./datasets/" + datasets + "/testB/" + mid + ".png")
        st = i + " PSNR: " + str(ms) + " SSIM: " + str(ssi) + "\n\n"
    psnr_all.append(ms)
    ssim_all.append(ssi)
    print(st)
    f.write(st)

print(np.mean(psnr_all), np.mean(ssim_all))
f.write("PSNR: " + str(np.mean(psnr_all)))
f.write("\nSSIM: " + str(np.mean(ssim_all)))

f.close()
