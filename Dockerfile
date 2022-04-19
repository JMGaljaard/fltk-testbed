FROM ubuntu:20.04

MAINTAINER Jeroen Galjaard <J.M.Galjaard-1@student.tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# ENV GLOO_SOCKET_IFNAME=eth0
# ENV TP_SOCKET_IFNAME=eth0

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository and fetch packages
RUN apt-get update \
    && apt-get install -y python3.9

# Setup pip3.9
RUN apt install -y curl python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py
# Add Pre-downloaded models (otherwise needs be run every-time)
ADD data/ data/

# Use cache for pip, otherwise we repeatedly pull from repository
COPY requirements-cpu.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,mode=0777 python3.9 -m pip install -r requirements.txt

# Add FLTK and configurations
ADD fltk fltk
ADD configs configs
ADD experiments experiments
