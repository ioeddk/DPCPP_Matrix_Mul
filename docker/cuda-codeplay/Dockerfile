#  docker build -t name .

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

COPY scripts/* /home/

RUN /home/prerequisite.sh
RUN /home/cuda-11-6.sh
RUN /home/dpcpp-codeplay.sh

RUN rm /home/*.sh /home/*.deb

RUN echo "export PATH=/home/llvm/build/bin:$PATH" >> /root/.bashrc
RUN echo "export LD_LIBRARY_PATH=/home/llvm/build/lib:$LD_LIBRARY_PATH" >> /root/.bashrc
# ADD nvcc bin to path!!!
