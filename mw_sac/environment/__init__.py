# C:\Users\lukas\OneDrive\Documents\Dolphin Emulator\StateSaves\GOWE69.s01
import numpy as np
import sys
import time
import threading
from typing import Tuple, Optional, Dict, Callable
import torch
import threading
import queue
import io
import subprocess
import cv2
from environment.courseprogress import process_frame
from math import log10
from config import ACTION_PIPE_LOCATION, CV2_VIDEO_INDEX, STEP_DURATION, PAUSE_COMMAND, UNPAUSE_COMMAND, RESET_COMMAND
#from ..config import ACTION_PIPE_LOCATION, CV2_VIDEO_INDEX, STEP_DURATION, PAUSE_COMMAND, UNPAUSE_COMMAND, RESET_COMMAND

current_progress_global = 0

class Request:
    def __init__(self, reset: bool, response_queue: queue.Queue, toggle_pause: bool, reset_car: bool = False):
        self.toggle_pause = toggle_pause
        self.reset = reset
        self.response_queue = response_queue
        self.reset_car = reset_car

class Response:
    def __init__(self, action_queue: queue.Queue, reward: float, state: np.array, old_state: np.array, done: bool = False):
        self.action_queue = action_queue
        self.reward = reward
        self.state = state
        self.old_state = old_state
        self.done = done

def do_reset():
    subprocess.call(RESET_COMMAND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def pause():
    subprocess.call(PAUSE_COMMAND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def unpause():
    subprocess.call(UNPAUSE_COMMAND, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def prepare_get_state():
    video = cv2.VideoCapture(CV2_VIDEO_INDEX)
    # Initialize course progress cnn
    # TODO: course_progress_net = 
    # Open video input
    states = [(np.array([]), int)] * 5
    def get_state():
        ret, frame = video.read()
        if ret:
            old_state = states.pop(0)
            state, progress, speed = process_frame(frame)
            state = (state, speed)
            states.append(state)
            return state, old_state, progress
        else:
            return None, None, None
    return get_state

apply_action_pipe = open(ACTION_PIPE_LOCATION, mode='ab', buffering=0)
def prepare_apply_action():

    def apply_action(action: np.array):
        # Accelerate/Brake/Reverse by manipulating the y coordinates of the C stick
        apply_action_pipe.write(bytes(f'SET C 0.5 {float(action[0])}\n', encoding='utf-8'))
        # Steer by manipulating the x coordinates of the MAIN stick
        apply_action_pipe.write(bytes(f'SET MAIN {float(action[1])} 0.5\n', encoding='utf-8'))

    def reset_car():
        apply_action_pipe.write(bytes(f'PRESS Y\n', encoding='utf-8'))
        time.sleep(0.05)
        apply_action_pipe.write(bytes(f'RELEASE Y\n', encoding='utf-8'))

    return apply_action, reset_car

def environment_thread(input_queue: queue.Queue):
    print("Starting environment thread")
    apply_action, reset_car = prepare_apply_action()
    get_state = prepare_get_state()

    do_reset()

    tick_duration_ns = STEP_DURATION
    tick_start = time.monotonic_ns()

    current_progress = 0.0
    current_reward = -0.1
    paused = False

    global current_progress_global
    
    while True:
        # Do work here
        #print(tick_start, time.monotonic_ns())
        #print("environment waiting for request")
        req: Request = input_queue.get(block=True)
        #print("environment received")

        if req.toggle_pause:
            if paused:
                unpause()
                paused=False
            else:
                pause()
                paused=True
            continue

        if req.reset_car:
            reset_car()
            continue

        if req.reset:
            current_progress = 0.0
            do_reset()
            # Wait until get state produces valid old states
            for _ in range(5):
                state, _, new_progress = get_state()
                current_progress_global = new_progress
                current_reward = log10((state[1]/6) + (1/6))
                if new_progress != current_progress:
                    current_reward += (1000 * (new_progress - current_progress))
                    current_progress = new_progress
                apply_action(np.array([-1.0, 0.5], dtype=float))
                tick_end = time.monotonic_ns()
                sleep_duration = tick_duration_ns - (tick_end - tick_start)
                if sleep_duration > 0:
                    time.sleep(sleep_duration/1000000000.0)
                pass

        state, old_state, new_progress = get_state()
        current_progress_global = new_progress

        #print("State:", state)
        #print("Old state:", old_state)

        #print("Diff in states", np.sum(np.abs(np.subtract(state[0], old_state[0]))))
        if not paused and np.sum(np.abs(np.subtract(state[0], old_state[0]))) == 0:
            print("state doesn't change despite the game not getting paused, attempting to unpause")
            unpause()

        current_reward = log10((state[1]/6) + (1/6))
        if new_progress != current_progress:
            current_reward += (1000 * (new_progress - current_progress))
            current_progress = new_progress

        # Turn the state into an array
        state = np.append(state[0], (state[1] / 300)).astype(np.float32)
        old_state = np.append(old_state[0], (state[1] / 300)).astype(np.float32)
        
        action_queue = queue.Queue()
        #print("Environment thread putting response", req.response_queue)
        req.response_queue.put(Response(action_queue, current_reward, state, old_state))
        #print("Environment thread put response")

        #print("Environment thread waiting for action")
        action = action_queue.get(block=True)
        print('Action:', action)
        #print("Environment thread received action")
        apply_action(action)

        tick_end = time.monotonic_ns()
        sleep_duration = tick_duration_ns - (tick_end - tick_start)
        if sleep_duration > 0:
            time.sleep(sleep_duration/1000000000.0)
        # We use the expected tick start instead on monotonic_ns to ensure
        # that ticks last tick_duration_ns on average.
        # If a tick was short the next one will have bigger sleep duration and
        # if a tick was long the next one will have smaller sleep duration
        tick_start = tick_end + sleep_duration



class Environment():
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=environment_thread, args=(self.input_queue,), daemon=True)
        self.thread.start()

    def reset(self) -> queue.Queue:
        return self.step(reset=True)

    def step(self, reset=False) -> queue.Queue:
        response_queue = queue.Queue(maxsize=1)
        #print("Created response queue", response_queue)
        self.input_queue.put(Request(reset, response_queue, False))
        return response_queue
    
    def toggle_pause(self):
        self.input_queue.put(Request(False, None, True))

    def reset_car(self):
        self.input_queue.put(Request(False, None, False, True))

    def get_progress(self):
        global current_progress_global
        return current_progress_global