#include <CL/sycl.hpp>
#include <iostream>
#include "mmpy_kernel.cu"

using namespace cl::sycl;

// #define A(i, j, ld) A[(i) * (ld) + (j)]
// #define B(i, j, ld) B[(i) * (ld) + (j)]
// #define out(i, j, ld) out[(i) * (ld) + (j)]

// tilescale (# of points computed by each thread)
#define TILESCALE_M 8
#define TILESCALE_N 4
#define TILESCALE_K 4

#define TILEDIM_M 128
#define TILEDIM_N 64

#define TILEDIM_K 64

constexpr int n=8;

// CUDA device selector
class CUDASelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    if(device.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda){
      std::cout << " CUDA device found " << std::endl;
      return 1;
    } else{
      return -1;
    }
  }
};

int main() {
    
    auto R_A = range<1>(n*n);
    auto R_B = range<1>(n*n);
    auto R_C = range<1>(n*n);
    std::vector<int> v_A(n*n,1);
    std::vector<int> v_B(n*n,1);
    std::vector<int> v_C(m*n,0);
    
    device dev{CUDASelector().select_device()};
    context myContext{dev};
    queue Q{myContext, dev};
    // sycl::queue Q{CUDASelector};
    
    {
        // Initialize Buffer
        buffer<int, 1> A_buf(v_A.data(), R_A);
        buffer<int, 1> B_buf(v_B.data(), R_B);
        buffer<int, 1> C_buf(v_C.data(), R_C);

        Q.submit([&](handler& h) {
                // stream out_prt(1024, 256, h);  // What are those two parameters are used for? 
            
                // Access to the write buffer
                auto accA = A_buf.get_access<access::mode::read>(h);
                auto accB = B_buf.get_access<access::mode::read>(h);
                auto accC = C_buf.get_access<access::mode::read_write>(h);
        
                h.host_task([=](interop_handle ih) {  // The should corresponding to the m rows and m columns of the output
                    auto A = reinterpret_cast<double*>(ih.get_native_mem<backend::ext_oneapi_cuda>(accA));
                    auto B = reinterpret_cast<double*>(ih.get_native_mem<backend::ext_oneapi_cuda>(accB));
                    auto C = reinterpret_cast<double*>(ih.get_native_mem<backend::ext_oneapi_cuda>(accC));
                    
                    dim3 grideSize{ (n % TILEDIM_N) == 0 ? (n / TILEDIM_N) : (n / TILEDIM_N + 1), (n % TILEDIM_M) == 0 ? (n / TILEDIM_M) : (n / TILEDIM_M + 1), 1};
                    dim3 blockSize{TILEDIM_N/TILESCALE_N, TILEDIM_M/TILESCALE_M, 1};

                    vecAdd<<<gridSize, blockSize, 64 * 1024>>>(n, C, A, B);  // but here how to define multidimensional array? 
                    // Interop with host_task doesn't add CUDA event to task graph
                    // so we must manually sync here.
                    cudaDeviceSynchronize();
                    // out_prt << idx << endl;
                }); 
                auto result = C_buf.get_access<access::mode::read>(h); // synchrounize the outputs. 
        });
        

        // Should I use accessor to access the result matrix or the original vector itself. 
        auto result = out_buf.get_access<access::mode::read>(); // synchrounize the outputs. 
        for (int i=0; i<m; i++){
            for (int j = 0; j < n; j++){
                std::cout << result[i*k+j] << " ";
            }
            std::cout << "\n";
        }
    }
    

    return 0;
}
