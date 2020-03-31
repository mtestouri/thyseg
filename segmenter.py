import os
import numpy as np
import cv2

class Segmenter:
    def train(self, x_train, y_train, model_file=None):
        raise NotImplementedError

    def predict(self, x_test, model_file=None):
        raise NotImplementedError

    def segment(self, x_test, y_test=None, model_file=None):
        # compute predictions
        y_predicts = self.predict(x_test, model_file)
        # write the image files
        if not os.path.exists('segmentations'):
            os.makedirs('segmentations')
        im_h = x_test.shape[1]
        im_w = x_test.shape[2]
        z = np.zeros((im_h, im_w, 1), dtype=np.uint8)
        sep = np.ones((im_h, 10, 3), dtype=np.uint8)*255
        for i in range(len(x_test)):
            # recreate image masks from classes
            y_test_i = np.concatenate(
                    (z, y_test[i].reshape(im_h, im_w, 1)*180, z), axis=2)
            y_predicts_i = np.concatenate(
                    (z, y_predicts[i].reshape(im_h, im_w, 1)*180, z), axis=2)
            # write files
            img = np.concatenate((x_test[i], sep,
                                 x_test[i] + y_test_i, sep,
                                 x_test[i] + y_predicts_i, sep,
                                 y_test_i, sep,
                                 y_predicts_i
                                 ), axis=1)
            cv2.imwrite("segmentations/seg" + str(i) + ".jpg", img)