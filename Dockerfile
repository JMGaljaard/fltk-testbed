FROM bitnami/pytorch:1.12.1

MAINTAINER Jeroen Galjaard <J.M.Galjaard-1@student.tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Add Pre-downloaded models (otherwise needs be run every-time)
ADD data/ data/

# Use cache for pip, otherwise we repeatedly pull from repository

# Make type of docker image comatible
ARG req_type
COPY requirements-${REQUIREMENT_TYPE:-cpu}.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,mode=0777 python3 -m pip install -r requirements.txt

# Add FLTK and configurations
ADD fltk fltk
ADD configs configs
ADD experiments experiments
