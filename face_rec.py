import sys, os, cv2, time, csv, numpy as np
import math
import dlib
import glob
from skimage import io
from scipy.spatial import distance
from os import listdir


def read_CSV(labels, img_urls, path):
    lists = os.listdir('./faces')
    for line in lists:
        if (line[len(line) - 3:] == 'jpg'):
            label = line[0:len(line) - 4]
            img_urls.append('./faces/' + line)
            labels.append(label)


def euklid(A, B):
    euklid = 0
    for i in range(0, len(A)):
        euklid = (A[i] - B[i]) ** 2

    return math.sqrt(euklid)


def load_vectors(path, detector, sp, facerec):
    img = io.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(gray)

    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor_etalon = facerec.compute_face_descriptor(img, shape)

        return face_descriptor_etalon


def main():
    faces_path = './db.CSV'

    face_rec_model_path = './models/dlib_face_recognition_resnet_model_v1.dat'
    predictor_path = './models/shape_predictor_68_face_landmarks.dat'

    thresold = 0.55

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    labels = []
    img_urls = []

    etalon_vector = []
    read_CSV(labels, img_urls, faces_path)

    for path in img_urls:
        etalon_vector.append(load_vectors(path, detector, sp, facerec))

    dir = "./"
#    while True:
    files = listdir(dir)
    for f in files:
        if (f[len(f) - 3:] == "png"):
            try:
                time.sleep(0.75)
                img = io.imread(dir + f)
                os.remove(dir + f)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                dets = detector(gray)
                for k, d in enumerate(dets):
                    shape = sp(gray, d)
                    face_descriptor = facerec.compute_face_descriptor(gray, shape)

                    predict_name = 'NoName'

                    for i in range(0, len(labels)):
                        dist = distance.euclidean(etalon_vector[i], face_descriptor)
                        if (dist < thresold):
                            predict_name = labels[i]

                    cv2.putText(gray, predict_name, (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 2,
                                cv2.LINE_AA)
                    cv2.rectangle(gray, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1, 8, 0)
                fn = f[0:len(f) - 3] + 'jpg'
                cv2.imwrite(dir + fn, gray)

            except:
                False


main()
