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
        im_h = x_test.shape[2]
        im_w = x_test.shape[1]
        z = np.zeros((im_h, im_w, 1), dtype=np.uint8)
        sep = np.ones((im_h, 10, 3), dtype=np.uint8)*255
        for i in range(len(x_test)):
            # recreate image masks from classes
            y_test_i = np.concatenate(
                (z, y_test[i, :, :, 1].reshape(im_h, im_w, 1)*180, z), axis=2)
            y_pred_i = np.concatenate(
                (z, y_predicts[i, :, :, 1].reshape(im_h, im_w, 1)*180, z), axis=2)
            # combine masks and images
            sup_y_i = x_test[i] + y_test_i
            sup_y_pred_i = x_test[i] + y_pred_i
            # add labels
            cv2.putText(y_test_i, "manual", (10, 30), fontScale=1, thickness=2, 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
            cv2.putText(y_pred_i, "predicted", (10, 30), fontScale=1, thickness=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))
            # write files
            img = np.concatenate((x_test[i], sep,
                                 sup_y_i, sep,
                                 sup_y_pred_i, sep,
                                 y_test_i, sep,
                                 y_pred_i), axis=1)
            cv2.imwrite("segmentations/seg" + str(i) + ".jpg", img)