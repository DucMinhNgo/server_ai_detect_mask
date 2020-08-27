# import library use for redis
import redis
import json
import time
from datetime import datetime

HOST_NAME="0.0.0.0"
PORT="6378"
PASSWORD=''


r = redis.Redis(
    host=HOST_NAME,
    port=PORT,
    password=PASSWORD,
)

# k-nearest neighbors
import numpy as np
import cv2 as cv
import os
import argparse
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors

# import mtcnn Library
from mtcnn.mtcnn import MTCNN
import cv2
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

class DatasetLoader:
    def load(self, images_paths, verbose = -1):
        # initial list of images and labels
        data = []
        labels = []
        # loop input path and read data
        for (i, path) in enumerate(images_paths):
            # load image
            image = cv.imread(path)
            label = path.split(os.path.sep)[-2]

            # resize image
            image = cv.resize(image, (32, 32))

            # push into data list
            data.append(image)
            labels.append(label)

            # Show update
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))
        return (np.array(data), np.array(labels))

NAME_QUEUE='queue'
print ('Process is starting')
if __name__ == "__main__":
    image_paths = list(paths.list_images("./dataset/"))
    print (len(image_paths))
    sdl = DatasetLoader()
    (dataset, labels) = sdl.load(image_paths, verbose=50)
    # dataset.shape[0] = 3000

    dataset = dataset.reshape((dataset.shape[0], 32*32*3))
    # Show memory consumption
    print("[INFO] features matrix: {:.1f}MB".format(
        dataset.nbytes / (1024 * 1024.0)
    ))
    # print (labels)
    # Encode labels as intergers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # print (dataset[:1, :].shape)
    # print (labels)
    # Partition the data
    # Training: 75%, test: 25%
    (X_train, X_test, y_train, y_test) = train_test_split(dataset, labels, test_size=0.25, random_state=42)

    clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2, weights = 'distance')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print ('predict label: ')
    print (y_pred)
    print ('ground true: ')
    print (y_test)
    _percent_predict = accuracy_score(y_test, y_pred)*100
    print (_percent_predict)

    # create detector
    detector = MTCNN()
    # config text for image
    # front
    front = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale 
    fontScale = 1
    # color
    color = (0,255,0)
    # Line thickness of 2 px 
    thickness = 2
    while True:
        while(r.llen(NAME_QUEUE)!=0):
            _data = r.lpop(NAME_QUEUE)
            _result = json.loads(_data)
            print (_result)
            print (_result["input"])
            _path_input = _result["input"]
            image_input = cv2.imread(_path_input)
            image = cv2.resize(image_input, (960, 540))
            cv2.imwrite(_result["outputResize"], image)
            result = detector.detect_faces(image)
            _image_name = _path_input.split(os.path.sep)[-1]
            _name_split = _image_name.split('.')
            # print (_name_split[0])
            # print (_name_split[1])
            pre_name = _name_split[0]
            tail_name = _name_split[1]
            _len_result = len(result)
            _num_face = 0
            for person in result:
                _label_name = pre_name + "_" + str(_num_face) + "." + tail_name
                # _label_name = str(_num_face) + "." + tail_name
                print ("_num_face: ", _label_name)
                _num_face += 1
                _data_test = []
                bounding_box = person['box']
                keypoints = person['keypoints']
                # init value for confidence
                _confidence = "0%"
                _confidence = person['confidence']
                _confidence = str(float(int(_confidence*10000)/100)) + "%"
                print (_confidence)
                x1 = bounding_box[0]
                y1 = bounding_box[1]
                x2 = bounding_box[0] + bounding_box[2]
                y2 = bounding_box[1] + bounding_box[3]

                crop_img = image[y1: y2, x1:x2]
                # save image use for label
                print (_result["pathCropPiture"])
                print (_result["filename"])
                # _file_name.split('.')
                # cv2.imshow('image', crop_img)
                cv2.imwrite(_result["pathCropPiture"] + "/" + _label_name, crop_img)
                crop_img = cv2.resize(crop_img, (32, 32))
                _data_test.append(crop_img)
                if (len(_data_test) != 0):
                    _np_data_test = np.array(_data_test)
                    _np_data_test = _np_data_test.reshape((_np_data_test.shape[0], 32*32*3))
                    y_pred = clf.predict(_np_data_test)
                    """
                        define:
                            + with_mask: 0
                            + without_mask: 1
                    """
                    print ('result: ', str(y_pred[0]))
                    print ('percent: ', str(_percent_predict))
                    # add text to image
                    _num_result = y_pred[0]
                    org = (x1 - 50, y1)
                    if _num_result == 0:
                        _text_result = 'mask'
                        color = (0,255,0)
                    else:
                        _text_result = 'no mask'
                        color = (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    # rectangle detect face
                    image = cv2.putText(image, _text_result + ":" + _confidence, org, front, fontScale,
                                        color, thickness, cv2.LINE_AA, False)
                else:
                    print ('not found face in this picture ')
            cv2.imwrite(_result["output"], image)
            cv2.waitKey(0)
            _result["status"] = 'done'
            r.set(_result["idTask"], json.dumps(_result))
            # print (r.get(_result["idTask"]))
            print ('response')
            print (datetime.now())
    print ('Process is finish')