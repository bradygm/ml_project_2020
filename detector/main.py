from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Input, Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from detect_objects import find_moving_objects, format_for_network


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image_size = (640, 360)


class Pipeline():
    def __init__(self):
        self.cap = None
        self.model = None
        self.video_writer = None

    def setup(self, video_file, nn_model_file):
        # load video
        self.cap = cv2.VideoCapture(video_file)

        # load model
        self.model = keras.models.load_model(nn_model_file)

        # video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter('output.avi',fourcc, 20.0, image_size)

    def detect_target(self):
        last_frame = None
        is_first = True
        while (True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if frame is None:
                break
            
            frame = cv2.resize(frame, image_size)

            # skip first frame
            if is_first:
                last_frame = np.copy(frame)
                is_first = False
                continue

            frame_corrected, bboxes, mask = find_moving_objects(last_frame, frame, False, './')
            # print(mask.shape)
            # print(mask)
            if type(bboxes) == type(False):
                continue
            
            candidates = format_for_network(bboxes, frame_corrected)
            # candidates = rgb2gray(candidates)

            if candidates.size > 0:
                probabilities = self.model.predict(candidates)
                predictions = np.argmax(probabilities, axis=1)
                # print(candidates.shape, predictions.shape)

                for i in range(len(bboxes)):
                    # print(probabilities[i])

                    debug_text = None
                    if predictions[i] == 1:
                        debug_text = f"bird {probabilities[i][1]:.2f}"
                    if predictions[i] == 2:
                        debug_text = f"aircraft {probabilities[i][2]:.2f}"

                    if predictions[i] == 1 or predictions[i] == 2:
                        minr, minc, maxr, maxc = bboxes[i]
                        cv2.rectangle(frame_corrected, (minc, minr), (maxc, maxr), (0, 0, 255), 1)
                        cv2.putText(frame_corrected, debug_text, (minc, minr), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                    
                    minr, minc, maxr, maxc = bboxes[i]
                    cv2.rectangle(mask, (minc, minr), (maxc, maxr), (0, 0, 255), 1)

            # Display the resulting frame
            cv2.imshow('frame_corrected', frame_corrected)
            cv2.imshow('blob_mask', mask)
            self.video_writer.write(frame_corrected)

            last_frame = np.copy(frame)
            cvk = cv2.waitKey(1)
            if cvk & 0xFF == ord('q'):
                break
            else:
                continue

    def predict(self):
        pass

    def visualize(self):
        pass


def main():
    pipeline = Pipeline()
    pipeline.setup(sys.argv[1], sys.argv[2])
    pipeline.detect_target()


if __name__ == "__main__":
    main()