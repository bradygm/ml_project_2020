import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from detector.detect_objects import find_moving_objects


class Pipeline():
    def __init__(self):
        self.cap = None    

    def load_video(self, filename):
        # load video
        self.cap = cv2.VideoCapture(filename)

    def detect_bbox(self):
        last_frame = None
        is_first = True
        while (True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if frame is None:
                break
            
            # skip first frame
            if is_first:
                last_frame = np.copy(frame)
                is_first = False
                continue

            frame_corrected, bboxes = find_moving_objects(last_frame, frame, False, './')
            for bbox in bboxes:
                minr, minc, maxr, maxc = bbox
                cv2.rectangle(frame_corrected, (minc, minr), (maxc, maxr), (0, 0, 255), 1)

            # Display the resulting frame
            cv2.imshow('frame_corrected', frame_corrected)

            last_frame = np.copy(frame)
            cvk = cv2.waitKey(0)
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
    pipeline.load_video(sys.argv[1])
    pipeline.detect_bbox()


if __name__ == "__main__":
    main()