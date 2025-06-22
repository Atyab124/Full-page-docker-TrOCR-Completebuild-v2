# Base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install Base Packages
RUN apt-get update
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libx11-6
RUN apt install libxext6

# Run Sym-linking Protocol
RUN ln -s /opt/nvidia/nsight-compute/2022.2.1/host/linux-desktop-glibc_2_11_3-x64/Mesa/libGL.so.1 /usr/lib/libGL.so.1

RUN ln -s /opt/nvidia/nsight-compute/2022.2.1/host/linux-desktop-glibc_2_11_3-x64/libssl.so.1.1 /usr/lib/libssl.so.1.1

RUN ln -s /opt/nvidia/nsight-compute/2022.2.1/host/linux-desktop-glibc_2_11_3-x64/libcrypto.so.1.1 /usr/lib/libcrypto.so.1.1


# Set working directory
WORKDIR /home/trocr
COPY . /home/trocr

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose Port

EXPOSE 8080

CMD ["python3", "master_api_demo.py"]
