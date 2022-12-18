FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

COPY scripts/* /home/

RUN /home/prerequisite.sh
RUN /home/oneapi.sh
RUN /home/cuda-11-6.sh
RUN /home/oneapi-for-nvidia-gpus-2023.0.0-linux.sh

RUN rm /home/*.sh /home/*.deb

RUN echo ". /opt/intel/oneapi/setvars.sh --include-intel-llvm" >> /root/.bashrc
