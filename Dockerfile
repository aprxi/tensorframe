FROM nvidia/cuda:10.0-base as base
LABEL maintainer "TENSORFRAME.AI <anthony.potappel@gmail.com>"

RUN apt-get update --fix-missing \
    && apt-get -y install --no-install-recommends \
        cuda-curand-$CUDA_PKG_VERSION \
        python3 \
        python3-yaml \
        python3-six \
        supervisor

RUN apt-get -y install python3-pip \
    && python3 -m pip install numpy \
    && python3 -m pip install pyarrow \
    && python3 -m pip install jupyter \
    && python3 -m pip install tensorframe \
    && apt-get -y remove python3-pip \
    && rm -rf /var/lib/apt/lists/* \ 
    && apt-get -y autoremove --purge && apt-get clean

COPY files/entrypoint.sh /bin/entrypoint.sh
COPY files/supervisor_jupyter /etc/supervisor/conf.d/jupyter.conf
RUN chmod +x /bin/entrypoint.sh \
    && mkdir -p /service /etc/supervisor/conf.d /var/run/supervisor /var/run/nvidia-persistenced
