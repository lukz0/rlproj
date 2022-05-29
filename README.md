# RL project

## Download models and datasets from github releases
After downloading the tar archive extract it at the base of the repository using:
`tar -xvf largefiles.tar.gz`

## Currently only linux is supported because the project uses named pipes to communicate with the dolphin emulator
To run the project on windows the code sending actions to the 

## Dependencies Ubuntu 20.04
`sudo apt update && sudo apt install xdotool -y`

`sudo snap install dolphin-emulator --beta`

Open dolphin emulator, go to graphics settings and turn "Auto-Adjust Window Size" on

Install obs studio with virtual camera support according to https://obsproject.com/wiki/install-instructions

Open obs and go to File>Settings>Video and change base and output resolution to 640x480 (the dimensions are in the width x height order)

Then Add window input
Then right click on the input on the canvas, and select Transform>Fit To Screen

Then start a virtual camera with the "Start Virtual Camera" button (as mentioned in the obs install link v4l2loopback-dkms is required for virtual camera support on ubuntu)

Acquire a nkit version of need for speed: most wanted

Start up the dolphin emulator and load the save from the file "69-GOWE-NFSMWBOT.gci" using Tools>Memory Card Manager (You may need to create a memory card with the filename MemoryCardA.USA.raw before importing the save)

Go to quick race, select porsche carrera gt as the car and press "Ctrl + F6" to create a quicksave that will be used to reset the environment

Set up pipe input according to: https://wiki.dolphin-emu.org/index.php?title=Pipe_Input and then change the ACTION_PIPE_LOCATION variable in the file mw_sac/config/\_\_init\_\_.py

Install pytorch for your current cuda version (cuda is required to run this project)

Install other dependencies:
`pip3 install --user numpy opencv-python`

## Running the project
The project can be configured using the `mw_sac/config/__init__.py`

The project can be ran by: `python3 mw_sac`
