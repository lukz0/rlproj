import random
import sqlite3
import cv2
from functools import partial
from typing import Iterator
import numpy as np
import sys
import time

cap = cv2.VideoCapture(int(sys.argv[1]))

# 500 ms
tick_duration_ns = 500000000
tick_start = time.monotonic_ns()

TABLE_NAME = 'whole_frames2'

with sqlite3.connect('frames.sqlite3') as db:
    db.execute(f'CREATE TABLE IF NOT EXISTS {TABLE_NAME} (image BLOB NOT NULL);')

    # loop over
    while True:
        #print('loop start')
        # read frames from stream
        #frame = stream.read()
        ret, vidframe = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', vidframe)
        
        db.execute(
            f'INSERT INTO {TABLE_NAME} (image) VALUES(?)',
            (
                bytes(cv2.imencode('.png', vidframe)[1]),
            )
        )


        tick_end = time.monotonic_ns()
        sleep_duration = tick_duration_ns - (tick_end - tick_start)
        sleep_duration_ms = sleep_duration//1000000
        if sleep_duration_ms <= 0: sleep_duration_ms = 1
        print(sleep_duration_ms)

        # check for 'q' key if pressed
        key = cv2.waitKey(delay=sleep_duration_ms) & 0xFF
        #print('wait_end')
        tick_start = tick_end + sleep_duration
        if key == ord("q"):
            break