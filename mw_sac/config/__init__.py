import argparse

# The location of the fifo/named pipe used to send input to the emulator, see:
# https://wiki.dolphin-emu.org/index.php?title=Pipe_Input
ACTION_PIPE_LOCATION='/home/lukz0/.local/share/dolphin-emu/Pipes/pipe1'

# Filenames
REPLAY_BUFFER_FILENAME = 'replay_buffer2.npz'
SAC_FILENAME = 'sac2.pt'
LOGFILE = 'log4.txt'

# The index of the virtual camera device created by obs
# The canvas in obs should have the dimensions 640x480
CV2_VIDEO_INDEX=0

# The duration of each step in nanoseconds
STEP_DURATION = 100000000

# Commands used for managing the environment
PAUSE_COMMAND = ['xdotool', 'getactivewindow', 'key', 'F10']
UNPAUSE_COMMAND = ['xdotool', 'getactivewindow', 'key', 'F10']
RESET_COMMAND = ['xdotool', 'getactivewindow', 'key', 'F6']

# Input and output size
INPUT_SIZE = 16*2 + 2 # latent space + old latent space + speed + old speed
OUTPUT_SIZE = 2


EVAL = False
# Set to 0 to initialize new networks
# Set to the next episode number to load existing networks
START_FROM_EPISODE = 0

# # Hyperparameters
# HIDDEN_SIZE = 256
# STEPS_PER_EPISODE = 3000
# EPISODES = 100
# UPDATE_AFTER = 1000
# UPDATE_EVERY = 50
# REPLAY_BUFFER_SIZE = 250000
# POLYAK = 0.995
# ALPHA = 0.2
# LEARNING_RATE = 1e-3
# BATCH_SIZE = 32

# Hyperparameters
HIDDEN_SIZE = 256
STEPS_PER_EPISODE = 3000
EPISODES = 100
UPDATE_AFTER = 1000
UPDATE_EVERY = 50
REPLAY_BUFFER_SIZE = 250000 * 8
POLYAK = 0.995
ALPHA = 0.2
LEARNING_RATE = 1e-4
BATCH_SIZE = 512