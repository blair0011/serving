#if NUANCE_CUDA

#define EIGEN_USE_GPU

//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "downsample_grad.h"

using namespace tensorflow;
typedef Eigen::GpuDevice GPUDevice;

#define EIGEN_USE_GPU

#define IDX2R(i,j,ld) (((i)*(ld))+(j))

#define TILE_WIDTH 16

// Define the CUDA kernel.
template <typename T>
__global__ void DownsampleSentenceMaxGradCudaKernel(const int input_d0, const int input_d1,
        const int offset_size, const T * __restrict__ input, const int64 * __restrict__ offset,
        const T * __restrict__ input_gradient, T * __restrict__ output_gradient) {
    //output: [offset_size-1][input_d1]
    if (blockIdx.x > 0) return;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    // Checking must be performed over the input
    if (e >= input_d0) return;
    if (c >= input_d1) return;

    int start = 0, steps = 0, end = 0, half_point = 0, ntotal_threads = 0, init_pos = 0;
    __shared__ T max[TILE_WIDTH][TILE_WIDTH];
    __shared__ int argmax[TILE_WIDTH][TILE_WIDTH];


    // We're going to compute the maximums per subsequence (from offset[i] to offset[i+1])
    // for all the filters at the same time
    for(int i = 0; i < offset_size-1; ++i){
        start = offset[i];
        end = offset[i+1];

        steps = (end-start)/blockDim.x + 1;
        init_pos = threadIdx.x+start;

        // Load into shared memory the maximums and indexes 
        // First initialize shared memory initial indexes and the lowest possible values
        max[threadIdx.x][threadIdx.y] = std::numeric_limits<T>::lowest();
        argmax[threadIdx.x][threadIdx.y] = 0;
        // If the thread is not out of the subsequence, fill the shared memory
        if (init_pos < end){
            max[threadIdx.x][threadIdx.y] = input[IDX2R(init_pos, c, input_d1)];
            argmax[threadIdx.x][threadIdx.y] = init_pos;
            // If the subsequence length is larger than the number of threads ...
            for (int s = 1; s < steps; ++s){
                if (init_pos+s*blockDim.x < end && 
                        max[threadIdx.x][threadIdx.y] < input[IDX2R(init_pos+s*blockDim.x, c, input_d1)]){
                    max[threadIdx.x][threadIdx.y] = input[IDX2R(init_pos+s*blockDim.x, c, input_d1)];
                    argmax[threadIdx.x][threadIdx.y] = init_pos+s*blockDim.x;
                }
            }
        }

        // Get the maximum values and its corresponding indexes from shared memory
        ntotal_threads = min(blockDim.x, input_d0);
        __syncthreads();
        while (ntotal_threads > 1){
            half_point = ((1+ntotal_threads) >> 1);
            if (threadIdx.x < half_point){
                if (threadIdx.x+half_point < ntotal_threads &&
                        max[threadIdx.x][threadIdx.y] < max[threadIdx.x+half_point][threadIdx.y]){
                    max[threadIdx.x][threadIdx.y] = max[threadIdx.x+half_point][threadIdx.y];
                    argmax[threadIdx.x][threadIdx.y] = argmax[threadIdx.x+half_point][threadIdx.y];
                }
            }
            __syncthreads();
            ntotal_threads = ((1+ntotal_threads) >> 1);
        }
        __syncthreads();

        // Copy the input gradient to the corresponding position of the output_gradient
        // which will be determined by the index of the maximum (per filter)
        if (threadIdx.x == 0){
            //int r = argmax[IDX2R(0,c,input_d1)];
            output_gradient[IDX2R(argmax[0][threadIdx.y],c,input_d1)] = input_gradient[IDX2R(i,c,input_d1)];
        }
        __syncthreads();
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct DownsampleSentenceMaxGradFunctor<GPUDevice, T> {
    void operator()(const GPUDevice& d, const int input_d0, const int input_d1, const int offset_size,
            const T *input, const int64 *offset, const T *input_gradient, T *output_gradient) {
        // Launch the cuda kernel.
        //
        // See core/util/cuda_kernel_helper.h for example of computing
        // block count and thread_per_block count.
        dim3 numBlocks(ceil((float)input_d0 / TILE_WIDTH), ceil((float)input_d1 / TILE_WIDTH), 1);
        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
        cudaMemset(output_gradient, 0, sizeof(T)*input_d0*input_d1);
        // All threads from a block execute within the same SM. Max threadsPerBlock = 1024
        DownsampleSentenceMaxGradCudaKernel<T><<<numBlocks, threadsPerBlock>>>(input_d0, input_d1, offset_size, input, offset, input_gradient, output_gradient);
    }
};

// Instantiate functors for the types of OpKernels registered.
template struct DownsampleSentenceMaxGradFunctor<GPUDevice, float>;
//template struct DownsampleSentenceMaxFunctor<GPUDevice, int>;

#endif  // NUANCE_CUDA
