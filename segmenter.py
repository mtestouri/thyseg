import numpy as np
import cv2
import os

class Segmenter:
    def train(self, x_train, y_train, weights_file=None):
        raise NotImplementedError

    def predict(self, x_test, weights_file=None):
        raise NotImplementedError

    def segment(self, x_test, y_test=None, weights_file=None):
        # compute predictions
        y_predicts = self.predict(x_test, weights_file)
        # write the image files
        if not os.path.exists('segmentations'):
            os.makedirs('segmentations')
        for i in range(len(x_test)):
            # recreate masks from classes
            y_test[i] = y_test[i]*(0, 128, 0)
            y_predicts[i] = y_predicts[i]*(0, 128, 0)
            # xrite files
            img = np.concatenate((x_test[i], x_test[i] + y_test[i],
                                  x_test[i] + y_predicts[i]/np.max()), axis=1)
            cv2.imwrite("segmentations/seg" + str(i) + ".jpg", img)