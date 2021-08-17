# Base image to start with
FROM ubuntu:20.04
 
# Who maintains this DockerFile
MAINTAINER Bart Cox <b.a.cox@tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Set required environmental variables for the working setup.
ENV GLOO_SOCKET_IFNAME=eth0
ENV TP_SOCKET_IFNAME=eth0

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository
RUN apt-get update \
  && apt-get install -y vim curl python3 python3-pip net-tools iproute2

#COPY data/ ./data
#COPY default_models ./default_models
# Copy the current folder to the working directory
ADD setup.py requirements.txt ./

# Use cache for pip, otherwise we repeatedly pull from repository
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt

ADD configs configs

ADD fltk fltk


# Install newest version of library
RUN python3 -m setup install

# Expose the container's port to the host OS
EXPOSE 5000


# Update relevant runtime configuration for experiment
COPY cloud_configs/cloud_experiment.yaml configs/cloud_config.yaml