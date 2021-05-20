# Base image to start with
FROM ubuntu:20.04
 
# Who maintains this DockerFile
MAINTAINER Bart Cox <b.a.cox@tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

ENV GLOO_SOCKET_IFNAME=eth0
ENV TP_SOCKET_IFNAME=eth0

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository
RUN apt-get update \
  && apt-get install -y vim curl python3 python3-pip net-tools iproute2

# Copy the current folder to the working directory
ADD setup.py requirements.txt ./
RUN python3 -m pip install -r requirements.txt

ADD configs configs

# Install all required packages for the generator
ADD fltk fltk
# Install newest version of library
RUN python3 -m setup install



# Expose the container's port to the host OS
EXPOSE 5000

# Run command by default for the executing container
# CMD ["python3", "/opt/Generatrix/rpc_parameter_server.py", "--world_size=2", "--rank=0", "--master_addr=192.168.144.2"]

#CMD python3 /opt/federation-lab/rpc_parameter_server.py --world_size=$WORLD_SIZE --rank=$RANK --master_addr=10.5.0.11
CMD python3 /opt/federation-lab/federated_learning.py $RANK $WORLD_SIZE 10.5.0.11