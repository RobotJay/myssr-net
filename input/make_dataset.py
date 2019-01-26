import numpy as np
import pandas as pd
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from TYY_utils import get_meta
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    root_path = "/home/heils-server/User/gwb/data/{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []
    data = pd.DataFrame(columns=['image_pth', 'label'])
    count = len(face_score)
    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        # img_pth = root_path + str(full_path[i][0])
        # img_age = age[i]
        # out_genders.append(int(gender[i]))
        # out_ages.append(age[i])
        img = cv2.imread(root_path + str(full_path[i][0]))
        if np.random.random() > 0.5:
            img = img[:, ::-1]

        if np.random.random() > 0.75:
            img = tf.contrib.keras.preprocessing.image.random_rotation(img, 20, row_axis=0, col_axis=1,
                                                                             channel_axis=2)
        if np.random.random() > 0.75:
            img = tf.contrib.keras.preprocessing.image.random_shear(img, 0.2, row_axis=0, col_axis=1,
                                                                          channel_axis=2)
        if np.random.random() > 0.75:
            img = tf.contrib.keras.preprocessing.image.random_shift(img, 0.2, 0.2, row_axis=0, col_axis=1,
                                                                          channel_axis=2)
        if np.random.random() > 0.75:
            img = tf.contrib.keras.preprocessing.image.random_zoom(img, [0.8, 1.2], row_axis=0, col_axis=1,
                                                                         channel_axis=2)

        pth = str(full_path[i][0]).replace('/','')
        cv2.imwrite('/home/heils-server/User/gwb/data/new_wiki/'+pth,img)
        data = data.append(
            pd.DataFrame({'image_pth': pth, 'label': [age[i]]}),
            ignore_index=True)
    data.to_csv('wiki_age_new.csv', index=None, columns=None)

    # np.savez(output_path,image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages), img_size=img_size)


def augment_data(images):
    for i in range(0, images.shape[0]):

        if np.random.random() > 0.5:
            images[i] = images[i][:, ::-1]

        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_rotation(images[i], 20, row_axis=0, col_axis=1,
                                                                             channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_shear(images[i], 0.2, row_axis=0, col_axis=1,
                                                                          channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_shift(images[i], 0.2, 0.2, row_axis=0, col_axis=1,
                                                                          channel_axis=2)
        if np.random.random() > 0.75:
            images[i] = tf.contrib.keras.preprocessing.image.random_zoom(images[i], [0.8, 1.2], row_axis=0, col_axis=1,
                                                                         channel_axis=2)

    return images

if __name__ == '__main__':
    main()
