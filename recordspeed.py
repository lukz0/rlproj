import random
import sqlite3
import cv2
from cv2 import VideoCapture
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision
from typing import Iterator
import numpy as np
import torch.nn.functional as F
#from vidgear.gears import ScreenGear
import cv2
import win32gui
import sys

print("selected camera:", sys.argv[1])

cap = cv2.VideoCapture(int(sys.argv[1]))

# loop over
with sqlite3.connect('frames.sqlite3') as db:
    db.execute('CREATE TABLE IF NOT EXISTS speed (image BLOB NOT NULL);')
    while True:
        # read frames from stream
        #frame = stream.read()
        ret, vidframe = cap.read()
        if not ret:
            break

        # check for frame if Nonetype
        # if frame is None:
        #     break

        # {do something with the frame here}
        cv2.imshow('Frame', vidframe[380:400, 515:563])
        img = bytes(cv2.imencode('.png', vidframe[380:400, 515:563], [cv2.IMWRITE_PNG_COMPRESSION, 0])[1])
        db.execute('INSERT INTO speed (image) VALUES(?)', (img,))
        # Show output window
        #cv2.imshow("Output Frame", vidframe[50:114, 556:620, :])
        #cv2.imshow("Output Frame", vidframe)
        #print(vidframe.shape)
        # print(vidframe.shape)

        # check for 'q' key if pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# close output window
cv2.destroyAllWindows()

# safely close video stream
# stream.stop()
cap.release()
cv2.destroyAllWindows()