from __future__ import print_function
import sys, os
import numpy as np
import cv2
from data import image_cols, image_rows, load_test_ids
from train import Learner
from skimage.io import imsave
from metric import np_dice_coef

def prep(img):
    img = img.astype('float32')
    img = cv2.resize(img, (image_cols, image_rows)) 

    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img

def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    imgs_id_test = load_test_ids()
    
    print ('Loading test_mask_res from %s' % Learner.test_mask_res)
    imgs_test = np.load(Learner.test_mask_res)
    print ('Loading imgs_exist_test from %s' % Learner.test_mask_exist_res)
    imgs_exist_test = np.load(Learner.test_mask_exist_res)
    print ('Loading imgs_exist_test from %s' % Learner.test_mask_gt)
    test_mask_gt = np.load(Learner.test_mask_gt)
    
    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    imgs_exist_test = imgs_exist_test[argsort]
    test_mask_gt = test_mask_gt[argsort]

    total = imgs_test.shape[0]
    print(total)
    print(imgs_test)
    ids = []
    rles = []
    mean_dice = 0
    each_pic_dice = 0
    for i in range(total):

        img = imgs_test[i, 0]
        print('prep', img.max())
        img_exist = imgs_exist_test[i]
        img_binary = prep(img)

        pred_dir = 'preds'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
        
        image = img_binary*255
        imsave(os.path.join(pred_dir, str(i) + '_pred.png'), image)
        print(test_mask_gt[i].shape, test_mask_gt[i].dtype)
        print(img_binary.shape, img_binary.dtype)

        test_mask_gt_img = cv2.resize(test_mask_gt[i, 0], (image_cols, image_rows)) 
        img_binary = img_binary.astype(np.float32)
        print(test_mask_gt_img.dtype, img_binary.dtype)
        each_pic_dice += np_dice_coef(test_mask_gt_img, img_binary)
        print(each_pic_dice)
        # for image, image_id in zip(test_masks, test_img_id):
            # image = (image[:, :, 0] * 255.).astype(np.uint8)
            # image = (image[0, :, :] * 255.)
        # new_prob = (img_exist + min(1, np.sum(img)/10000.0 )* 5 / 3)/2
        # if np.sum(img) > 0 and new_prob < 0.5:
        #     img = np.zeros((image_rows, image_cols))
        # rle = run_length_enc(img)
        # rles.append(rle)
        # ids.append(imgs_id_test[i])

        if i % 1000 == 0:
            print('{}/{}'.format(i, total))
    mean_dice = each_pic_dice / total
    print(mean_dice)
    # file_name = os.path.join(Learner.res_dir, 'submission.csv')

    # with open(file_name, 'w+') as f:
    #     f.write('img,pixels\n')
    #     for i in range(total):
    #         s = str(ids[i]) + ',' + rles[i]
    #         f.write(s + '\n')

def main():
    submission()


if __name__ == '__main__':
    sys.exit(main())
