#include <cuda_runtime.h>
#include <iostream>
#include <vector>


int main() {
  const size_t N = 1024;
  size_t size = N * sizeof(float);

  // host memory
  float* h_data = (float*)malloc(size);

  // device memory
  float* d_data = nullptr;
  cudaMalloc(&d_data, size);

  cudaError_t err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

  if(err != cudaSuccess) {
      std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
  } else {
      std::cout << "Copy succeeded!" << std::endl;
  }

  free(h_data);
  cudaFree(d_data);

  return 0;

}
