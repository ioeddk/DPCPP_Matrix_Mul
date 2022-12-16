#!/bin/bash

docker build -t ioeddk/dpcpp_cuda:latest ./cuda-codeplay;
#docker build -t dpcpp:cuda-intel ./cuda-intel > cuda-intel.log;
#docker build -t dpcpp:cuda-toolkit-codeplay ./cuda-toolkit-codeplay > cuda-toolkit-codeplay.log;
#docker build -t dpcpp:cuda-toolkit-intel ./cuda-toolkit-intel > cuda-toolkit-intel.log;

docker push ioeddk/dpcpp_cuda:latest;
