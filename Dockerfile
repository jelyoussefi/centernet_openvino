FROM ubuntu:24.10

ARG DEBIAN_FRONTEND=noninteractive

USER root

# Install system dependencies
RUN apt update -y && apt install -y \
    build-essential \
    software-properties-common \
    wget \
    gpg \
    python3-pip \
    python3-dev \
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
RUN apt install  -y   python3-setuptools 
RUN pip install --no-cache-dir --break-system-packages \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly openvino
RUN pip install --no-cache-dir --break-system-packages \
    nncf 

#RUN pip install --break-system-packages \
#	torch torchvision --index-url https://download.pytorch.org/whl/cpu


#--------------------------------------------------------------------------------------------------------------------------
# MMDeploy
#--------------------------------------------------------------------------------------------------------------------------
#RUN apt update -y && apt install -y --no-install-recommends \
#    software-properties-common build-essential wget gpg curl pciutils git cmake ninja-build \
#    g++ gcc unzip libtbb12 libgl1 libglib2.0-0 && \
#    rm -rf /var/lib/apt/lists/*

    
#RUN pip install --no-cache-dir --break-system-packages numpy \
#	openmim 'matplotlib<3.6.0' 'networkx>=2.8' \
#        mmengine 'mmcv>=2.0.0rc4,<2.2.0' 'mmdet>=3.0.0'

#RUN pip install --no-cache-dir --break-system-packages openvino==2024.6.0 

RUN PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install torch==2.2.0+cpu torchvision --break-system-packages
RUN pip install mmcv==2.1.* -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html --break-system-packages
RUN pip install openvino==2025.1 mmdet --break-system-packages
RUN apt install -y unzip

# ----------------------------------
# 5. Set Working Directory
# ----------------------------------
WORKDIR /workspace
