#include <CL/sycl.hpp>
#include <iostream>

// This attempt failed because the parallel_for structure cannot handel data dependency in different iterations. 
constexpr int k =2048;
constexpr int o = 1;
using namespace cl::sycl;

int main() {
    
    auto R_A = range<1>(k);
    auto R_B = range<1>(k);
    auto R_out = range<1>(o);
    std::vector<int> v_A(k,2);
    std::vector<int> v_B(k,2);
    std::vector<int> v_out(o,0);
    
    queue Q(cpu_selector{});
    
    {
        // Initialize Buffer
        buffer<int, 1> A_buf(v_A.data(), R_A);
        buffer<int, 1> B_buf(v_B.data(), R_B);
        buffer<int, 1> out_buf(v_out.data(), R_out);
        
        // Queue to submit the job.
        Q.submit([&](handler& h) {
            // Cannot use std::cout to print because SYCL runtime doesn't accept non constant variable. Use cl:sycl::stream instead. 
            stream out_prt(1024, 256, h);  // What are those two parameters are used for? 
            
            // Access to the write buffer
            auto A = A_buf.get_access<access::mode::read>(h);
            auto B = B_buf.get_access<access::mode::read>(h);
            auto out = out_buf.get_access<access::mode::read_write>(h);
            
            h.parallel_for(R_A, [=](id<1> idx) {  // The should corresponding to the m rows and m columns of the output
                out[0] = out[0] + A[idx]*B[idx];
                // out_prt << out[0] << endl;
            }); 
        });
        
        // Access to the buffer to synchronous the computation 
        auto result = out_buf.get_access<access::mode::read>();
        std::cout << result[0] << "\n";
        
    }
    return 0;
    

}
