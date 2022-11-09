#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

#define A(i, j, ld) A[(i) * (ld) + (j)]
#define B(i, j, ld) B[(i) * (ld) + (j)]
#define out(i, j, ld) out[(i) * (ld) + (j)]

constexpr int m=8;
constexpr int n=8;
constexpr int k=4;

int main() {
    
    auto R_A = range<1>(m*k);
    auto R_B = range<1>(k*n);
    auto R_out = range<1>(m*n);
    std::vector<int> v_A(m*k,3);
    std::vector<int> v_B(k*n,2);
    std::vector<int> v_out(m*n,0);
    
    queue Q(cpu_selector{});
    
    {
        // Initialize Buffer
        buffer<int, 1> A_buf(v_A.data(), R_A);
        buffer<int, 1> B_buf(v_B.data(), R_B);
        buffer<int, 1> out_buf(v_out.data(), R_out);
        
        for (int i_k = 0; i_k < k; i_k++) {
            // Queue to submit the job.
            Q.submit([&](handler& h) {
                stream out_prt(1024, 256, h);  // What are those two parameters are used for? 
            
                // Access to the write buffer
                auto A = A_buf.get_access<access::mode::read>(h);
                auto B = B_buf.get_access<access::mode::read>(h);
                auto out = out_buf.get_access<access::mode::read_write>(h);
        
                h.parallel_for(R_out, [=](id<1> idx) {  // The should corresponding to the m rows and m columns of the output
                    out[idx[0]] += A(idx[0] / k ,i_k,k) * B(i_k, idx[0] % k, k);
                    // out_prt << idx << endl;
                }); 
                auto result = out_buf.get_access<access::mode::read>(h); // synchrounize the outputs. 
            });
        }
        
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
