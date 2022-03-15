# Base image to start with
FROM ubuntu:20.04

MAINTAINER Bart Cox <b.a.cox@tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository
RUN apt-get update \
  && apt-get install -y vim curl python3 python3-pip net-tools iproute2

# Copy the current folder to the working directory
COPY setup.py ./
COPY requirements.txt ./

# Install all required packages for the generator
RUN python3 -m pip install -r requirements.txt

ENV GLOO_SOCKET_IFNAME=$NIC
ENV TP_SOCKET_IFNAME=$NIC

# Expose the container's port to the host OS
EXPOSE 5000

COPY fltk ./fltk
COPY configs ./configs

CMD python3 -m fltk remote $EXP_CONFIG $RANK --nic=$NIC --host=$MASTER_HOSTNAME $OPTIONAL_PARAMS