clear;
rm test.sh.e* test.sh.o*;
dpcpp -fsycl-unnamed-lambda mm.cpp -o mm;
./mm

#rm test.sh.e* test.sh.o*;
#dpcpp -fsycl-unnamed-lambda vm.cpp -o vm;
#./vm
