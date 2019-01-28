#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
from torch.autograd import Variable

import os

import numpy as np

import detect_face

from skimage import transform as trans
import cv2
import time

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

class Alignment:
    def __init__(self):

        image_size = (128, 128)
        self.image_size = image_size
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.9]
        self.factor = 0.85
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            self.sess = tf.Session()
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)

        # original src point
        # src = np.array([
        #     [30.2946, 51.6963],
        #     [65.5318, 51.5014],
        #     [48.0252, 71.7366],
        #     [33.5493, 92.3655],
        #     [62.7299, 92.2041]], dtype=np.float32)

        #transform [src * (192. / 112) + 32]
        src = np.array([
            [ 50.0 , 66.0],
            [80.0, 66.0],
            [65.0, 87.0],
            [ 53.0, 95.0],
            [77.0, 95.0]], dtype=np.float32)

        self.src = src

    def alignment(self, rimg, landmark,Debug=False):
        try:
            assert landmark.shape[0] == 68 or landmark.shape[0] == 5
            assert landmark.shape[1] == 2
        except Exception as e:
            print('No face and landmarks')
            return None
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
        if Debug:
            cv2.imshow('img',img)
            cv2.waitKey(0)
        return img

    def detect_max_face(self,img,Debug=False):
            _minsize = self.minsize
            _bbox = None
            _landmark = None
            bounding_boxes, points = detect_face.detect_face(img, _minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                bindex = 0
                if nrof_faces > 1:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    bindex = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T
                if Debug:
                    show = img.copy()
                    cv2.rectangle(show,(int(_bbox[0]),int(_bbox[1])),(int(_bbox[2]),int(_bbox[3])),color=(255,255,0))
                    for idx in range(_landmark.shape[0]):
                        cv2.circle(show,(_landmark[idx][0], _landmark[idx][1]),2,(0,255,255),2)
                    cv2.imshow('landmarks',show)
                    cv2.waitKey(0)

            return _bbox, _landmark

    def get_single_image(self,image_path):
        if not os.path.exists(image_path):
            return None
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
            return None
        else:
            if img.ndim < 2:
                print('Unable to align "%s", img dim error' % image_path)
                # text_file.write('%s\n' % (output_filename))
                exit()
            if img.ndim == 2:
                img = to_rgb(img)
            img = img[:, :, 0:3]
        return img

    def example(self,img_path1):
        img1 = self.get_single_image(img_path1)
        if img1 is None:
            print("read image failed!please check image source and path")
        bbox1, landmarks1 = self.detect_max_face(img1, Debug=False)

        face1 = self.alignment(img1, landmarks1, Debug=False)
        print("image size:",face1.shape)
        cv2.imshow('face1', face1)
        cv2.imshow('face roi', face1[32:256-32, 32:256-32,:])
        cv2.waitKey(0)

    def get_aligface(self, image):
        img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if img is None:
            return None
        bbox, landmarks = self.detect_max_face(img, Debug=False)
        face = self.alignment(img, landmarks, Debug=False)

        return face,bbox

if __name__ == "__main__":
    _alignment = Alignment()
    img_size = 92
    stage_num = [3,3,3]
    lambda_local = 0.25
    lambda_d = 0.25


    from MySSRNET92 import MySSRNet,MySSRNet_gen


    model_age = MySSRNet(stage_num, lambda_local, lambda_d)
    model_gen = MySSRNet_gen(stage_num, lambda_d, lambda_d)
    import torch

    resume_age = '/home/gwb/pycharm/project/SSR-Net-master/demo/log/asina/1492---[4.4812].pth'
    model_age.load_state_dict(torch.load(resume_age, map_location=lambda storage, loc: storage))
    model_age.eval()
    #
    # resume_gen = '/home/gwb/pycharm/project/SSR-Net-master/demo/log/imdb/gen/06/27---[0.2428].pth'
    # model_gen.load_state_dict(torch.load(resume_gen, map_location=lambda storage, loc: storage))
    # model_gen.eval()

    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        face, _bbox = _alignment.get_aligface(img)
        if face is None:
            continue
        face = cv2.resize(face, (img_size, img_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face = face.astype(np.float32)
        face = face / 255.0
        face = (face - mean) / std
        face = face.transpose(2, 0, 1)

        face = np.expand_dims(face,0)
        face = face.astype(np.float32)
        face_tensor = torch.from_numpy(face)
        face_variable = Variable(face_tensor)

        with torch.set_grad_enabled(False):
            predicted_ages = model_age(face_variable)
            # predicted_genders = model_gen(face_variable)

        age_str = str(round(predicted_ages.cpu().numpy()[0]))
        gender_str = 'male'
        # if predicted_genders[0]<0.5:
        #     gender_str = 'female'
        cv2.putText(img, age_str + "," , (int(_bbox[0]), int(_bbox[1])), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 1)
        print("age:\t", age_str,"gender:\t", )
        cv2.imshow('face', img)
        cv2.waitKey(1)
