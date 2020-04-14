# CUDA image to install NVIDIA libraries
#FROM nvidia/cuda as cuda
#FROM continuumio/anaconda3 as conda-base


#RUN pip install --no-cache-dir -r requirements.txt
#RUN conda install --file requirements.txt

# Installing anaconda solely to install rdkit because no pip install available


FROM tensorflow/tensorflow:1.15.0-gpu-py3 as tensor-base

RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    ca-certificates \
    build-essential \
    cmake \
    wget \
    libboost-dev \
    libboost-iostreams-dev \
    libboost-python-dev \
    libboost-regex-dev \
    libboost-serialization-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libcairo2-dev \
    libeigen3-dev \
    python3-dev \
    python3-numpy \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG RDKIT_VERSION=Release_2019_09_1
RUN wget --quiet https://github.com/rdkit/rdkit/archive/${RDKIT_VERSION}.tar.gz \
 && tar -xzf ${RDKIT_VERSION}.tar.gz \
 && mv rdkit-${RDKIT_VERSION} rdkit \
 && rm ${RDKIT_VERSION}.tar.gz

RUN mkdir /rdkit/build/
WORKDIR /rdkit/build

RUN cmake -Wno-dev \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr \
    -D Boost_NO_BOOST_CMAKE=ON \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D RDK_BUILD_AVALON_SUPPORT=ON \
    -D RDK_BUILD_CAIRO_SUPPORT=ON \
    -D RDK_BUILD_CPP_TESTS=OFF \
    -D RDK_BUILD_INCHI_SUPPORT=ON \
    -D RDK_BUILD_FREESASA_SUPPORT=ON \
    -D RDK_INSTALL_INTREE=OFF \
    -D RDK_INSTALL_STATIC_LIBS=OFF \
    ..

RUN make -j $(nproc) \
 && make install

RUN apt-get update && apt-get install -yq --no-install-recommends libxrender1
RUN apt-get update && apt-get install -yq --no-install-recommends libxext6

WORKDIR /root/
COPY . /root

# following is for shared libraries across python installations
# bc there is a previous python installation
# from debian
#ENV LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib/
#ENV PATH=$LD_LIBRARY_PATH:$PATH

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir results/
#RUN apt-get update \
# && apt-get install -yq --no-install-recommends \
#    libboost-atomic1.67.0 \
#    libboost-chrono1.67.0 \
#    libboost-date-time1.67.0 \
#    libboost-iostreams1.67.0 \
#    libboost-python1.67.0 \
#    libboost-regex1.67.0 \
#    libboost-serialization1.67.0 \
#    libboost-system1.67.0 \
#    libboost-thread1.67.0 \
#    libcairo2-dev \
#    python3-dev \
#    python3-numpy \
#    python3-cairo \
# && apt-get clean \
# && rm -rf /var/lib/apt/lists/*


