#!/usr/bin/env bash

# CD-ing into the correct directory
cd /home/$USER/fltk-testbed-gr-30

# Pull most recent changes
git pull

# Install the changes
sudo python3 setup.py install


