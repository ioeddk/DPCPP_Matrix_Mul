#  docker build -t name .

# FIXME: Want to change the installation path of the llvm folder to another place

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

COPY scripts/* /home/

RUN /home/prerequisite.sh
RUN /home/cuda-11-6.sh
RUN /home/dpcpp-codeplay.sh

RUN rm /home/*.sh /home/*.deb

RUN echo "export PATH=/home/llvm/build/bin:$PATH" >> /root/.bashrc
RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=/home/llvm/build/lib:$LD_LIBRARY_PATH" >> /root/.bashrc
