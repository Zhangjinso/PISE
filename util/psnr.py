gfla = '/home/jins/my_grade2/Global-Flow-Local-Attention/eval_results/fashion_new/'
adgan = '/home/jins/my_grade1/ADGAN/generated_imgs/img_crop'
bi = '/home/jins/my_grade1/BiGraphGAN/results_by_author/deepfashion_results/images_gen'
xing = '/home/jins/my_grade1/XingGAN/results/deepfashion_XingGAN/test_latest/images_gen'
patn = '/home/jins/my_grade1/Pose-Transfer_0.3/cvprresults/fashion_PATN_1.0/test_latest/gen'
pg = '/home/jins/mycode/PG-2020-posetransfer/results/fashion_PInet_PG/test_test/images_crop_m'

our_f = '/home/jins/my_grade3/CVPR21/eval_results/fashion_local_zero' #in

tar_f = '/home/jins/my_grade1/ADGAN/generated_imgs/img_tar'


import os
import cv2
from skimage.io import imread, imsave
import numpy as np
def addBounding(image, bound=40):
    h, w, c = image.shape
    image_bound = np.ones((h, w+bound*2, c))*255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound+w] = image

    return image_bound

 

def psnr_(orig, target, max_value):
    """Numpy implementation of PSNR."""
    mse = ((orig - target) ** 2).mean(axis=(-3, -2, -1))
    return mse, 20 * np.log10(max_value) - 10 * np.log10(mse)

    
def _main(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    #print(img2_path)
    img2 = cv2.imread(img2_path)
    if img2.shape[1]==256:
        img2 = img2[:,40:-40,:]
    psnr = psnr_(img1, img2, 255) 
    return psnr
 
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--folder", type=str, default="ou")
    args = parser.parse_args()


    score_list = []
    s_l = []
    a = os.listdir(tar_f) 
    for i in a:
        img2 = os.path.join(tar_f,i) 
        name = i
        if args.folder.startswith('ou'):         
            folder = our_f
        elif args.folder.startswith('bi'):
            folder = bi
        elif args.folder.startswith('g'):
            folder = gfla
        elif args.folder.startswith('ad'):
            folder = adgan
        elif args.folder.startswith('xing'):
            folder = xing
        elif args.folder.startswith('patn'):
            folder = patn
        elif args.folder.startswith('pg'):
            folder = pg

        img1 = os.path.join(folder,name)
        #os.rename(img1, os.path.join(folder, i))
        mse, psnr = _main(img1, img2)
        score_list.append(psnr)
        s_l.append(mse)
    print(len(score_list))
    print(args.folder+' psnr : '+ str(np.mean(score_list))+'  mse : '+str(np.mean(s_l)))
    
