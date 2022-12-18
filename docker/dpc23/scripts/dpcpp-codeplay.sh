git clone https://github.com/intel/llvm.git -b sycl;
cd llvm;
python3 ./buildbot/configure.py --cuda -t release --cmake-gen "Unix Makefiles";
cd build;
make sycl-toolchain -j `nproc`;
make install;
