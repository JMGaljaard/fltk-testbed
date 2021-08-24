FROM ubuntu:20.04

# Who maintains this DockerFile
MAINTAINER Jeroen Galjaard <J.M.Galjaard-1@student.tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables for GLOO and TP (needed for RPC calls)
ENV GLOO_SOCKET_IFNAME=eth0
ENV TP_SOCKET_IFNAME=eth0

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository and fetch packages
RUN apt-get update \
  && apt-get install -y vim curl python3 python3-pip net-tools iproute2

# Use cache for pip, otherwise we repeatedly pull from repository
ADD setup.py requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt

ADD configs configs
ADD fltk fltk
ADD scripts scripts

# Expose default port 5000 to the host OS.
EXPOSE 5000

# Update relevant runtime configuration for experiment
COPY configs/example_cloud_experiment.json configs/example_cloud_experiment.json