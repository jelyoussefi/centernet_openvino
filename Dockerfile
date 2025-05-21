FROM ubuntu:24.10

ARG DEBIAN_FRONTEND=noninteractive

USER root

# Install system dependencies
RUN apt update -y && apt install -y \
    build-essential \
    software-properties-common \
    wget \
    gpg \
    unzip \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-opencv \
    libopencv-dev \
    libqt5widgets5 \
    libtbb12 

# ----------------------------------
# 1. Install Intel Graphic Drivers
# ----------------------------------
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics
RUN apt install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
RUN apt install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
RUN apt install -y libze-dev intel-ocloc

# ----------------------------------
# 2. Install NPU Driver
# ----------------------------------
WORKDIR /tmp
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-fw-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    wget https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-level-zero-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb && \
    dpkg -i *.deb && \
    rm -f *.deb


# ----------------------------------
# 3. Install Python Dependencies
# ----------------------------------
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt --break-system-packages

# ----------------------------------
# 5. Set Working Directory
# ----------------------------------
WORKDIR /workspace
