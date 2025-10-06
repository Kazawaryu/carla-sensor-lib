FROM ubuntu:jammy

## Install Dependencies
RUN apt-get update && apt-get install -y \
    gfortran \
    wget \
    python3 \
    python3-pip \
    libtiff5 \
    libjpeg-dev \
    libpng16-16 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

#Set up the working directory
WORKDIR /home/CarlaCDASimAdapter/
ENV PYTHONPATH=/home/CarlaCDASimAdapter/src


# Copy the correct CARLA .whl for Python 3.10 from PythonAPI/carla
COPY PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl /tmp/
RUN pip3 install /tmp/carla-0.10.0-cp310-cp310-linux_x86_64.whl


RUN pip3 install numpy PyYAML scipy dataclasses

COPY ./src/ /home/CarlaCDASimAdapter/src
COPY ./config/ /home/CarlaCDASimAdapter/config