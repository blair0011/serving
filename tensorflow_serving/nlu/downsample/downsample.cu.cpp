#if NUANCE_CUDA

#define EIGEN_USE_GPU

//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "downsample.h"

using namespace tensorflow;
typedef Eigen::GpuDevice GPUDevice;

#define EIGEN_USE_GPU

#define IDX2R(i,j,ld) (((i)*(ld))+(j))

#define TILE_WIDTH 16

// Define the CUDA kernel.
template <typename T>
__global__ void DownsampleSentenceMaxCudaKernel(const int input_d0, const int input_d1,
        const int offset_size, const T * __restrict__ input, const int64 * __restrict__ offset, T * __restrict__ output) {
    if (blockIdx.x > 0) return;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    // Checking must be performed over the input
    if (e >= input_d0) return;
    if (c >= input_d1) return;

    // threadIdx.x and e will have the same value

    int start = 0, steps = 0, end = 0, half_point = 0,  ntotal_threads = 0, init_pos = 0;

    __shared__ T aux[TILE_WIDTH][TILE_WIDTH];

    // We're going to compute the maximums per subsequence (from offset[i] to offset[i+1])
    // for all the filters at the same time
    for(int i = 0; i < offset_size-1; ++i){
        start = offset[i];
        end = offset[i+1];

        steps = (end-start)/blockDim.x + 1;
        init_pos = threadIdx.x+start;

        // Load into shared memory the maximums
        // First initialize shared memory with the lowest possible values
        aux[threadIdx.x][threadIdx.y] = std::numeric_limits<T>::lowest();
        // If the thread is not out of the subsequence, fill the shared memory
        if (init_pos < end) {
            aux[threadIdx.x][threadIdx.y] = input[IDX2R(init_pos, c, input_d1)];
            // If the subsequence length is larger than the number of threads ...
            for (int s = 1; s < steps; ++s){
                if (init_pos+s*blockDim.x < end &&
                        aux[threadIdx.x][threadIdx.y] < input[IDX2R(init_pos+s*blockDim.x, c, input_d1)]){
                    aux[threadIdx.x][threadIdx.y] = input[IDX2R(init_pos+s*blockDim.x, c, input_d1)];
                }
            }
        }

        // Get maximum values from shared memory
        // ntotal_threads = blockDim.x;
        ntotal_threads = min(blockDim.x, input_d0);
        __syncthreads();
        while (ntotal_threads > 1){
            half_point = ((1+ntotal_threads) >> 1);
            // Each thread will compute the maximum between the values at threadIdx.x and
            // threadIdx.x+half_point and will copy the result to the initial position
            // (thread.Idx.x)
            if (threadIdx.x < half_point){
                if (threadIdx.x+half_point < ntotal_threads &&
                        aux[threadIdx.x][threadIdx.y] < aux[threadIdx.x+half_point][threadIdx.y]){
                    aux[threadIdx.x][threadIdx.y] = aux[threadIdx.x+half_point][threadIdx.y];
                }
            }
            __syncthreads();
            ntotal_threads = ((1+ntotal_threads) >> 1);
        }
        __syncthreads();

        // At the end, the global maximum will be at the first position
        // Copy the result back to global memory 
        if (threadIdx.x == 0){
            output[IDX2R(i,c,input_d1)] = aux[0][threadIdx.y];
        }
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct DownsampleSentenceMaxFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, const int input_d0, const int input_d1, const int offset_size,
            const T *input, const int64 *offset, T *output) {
        // Launch the cuda kernel.
        //
        // See core/util/cuda_kernel_helper.h for example of computing
        // block count and thread_per_block count.
        dim3 numBlocks(ceil((float)input_d0 / TILE_WIDTH), ceil((float)input_d1 / TILE_WIDTH), 1);
        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
        // All threads withing a block execute within the same SM. Max threadsPerBlock = 1024
        // DownsampleSentenceMaxCudaKernel<T><<<numBlocks, threadsPerBlock, sizeof(T)*TILE_WIDTH*input_d1>>>(input_d0, input_d1, offset_size, input, offset, output);
        DownsampleSentenceMaxCudaKernel<T><<<numBlocks, threadsPerBlock>>>(input_d0, input_d1, offset_size, input, offset, output);
    }
};

// Instantiate functors for the types of OpKernels registered.
template struct DownsampleSentenceMaxFunctor<GPUDevice, float>;
//template struct DownsampleSentenceMaxFunctor<GPUDevice, int>;

#endif  // NUANCE_CUDA
