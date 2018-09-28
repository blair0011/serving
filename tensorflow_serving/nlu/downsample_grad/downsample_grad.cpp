
#define  EIGEN_USE_THREADS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "downsample_grad.h"

#include <stdio.h>
#include <iostream>

#define IDX2R(i,j,ld) (((i)*(ld))+(j))

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DownsampleSentenceMaxGrad")
    .Attr("T: realnumbertype")
    .Input("input_matrix: T")
    .Input("offset: int64")
    .Input("input_grad: T")
    .Output("out_grad: T");

// CPU specialization of actual computation
template <typename T>
struct DownsampleSentenceMaxGradFunctor<CPUDevice, T>{
    void operator()(const CPUDevice &d, const int input_d0, const int input_d1, const int offset_size,
            const T *input, const int64 *offset, const T *input_gradient, T *output_gradient){

        // Alloc memory for auxiliary tensors; max and argmax
        T *max = new T[(offset_size-1)*input_d1];
        int *argmax = new int[(offset_size-1)*input_d1];

        int n_filter = input_d1;
        for (int e = 0; e < input_d0; ++e){
            for (int c = 0; c < input_d1; ++c){
                output_gradient[IDX2R(e,c,input_d1)] = 0;
            }
        }

        // Compute the maximum for the given offsets and the indexes where are located
        for(int e = 0; e < offset_size-1; ++e){
            int start = offset[e];
            int end = offset[e+1];
            if (end == start)
                continue;
            for (int c = 0;  c < n_filter; ++c){
                max[IDX2R(e,c,input_d1)] = input[IDX2R(start,c,input_d1)];
                argmax[IDX2R(e,c,input_d1)] = start;
            }
            for (int r = start+1; r < end; ++r){
                for (int c = 0; c < n_filter; ++c){
                    if (input[IDX2R(r,c,input_d1)] > max[IDX2R(e,c,input_d1)]) {
                        max[IDX2R(e,c,input_d1)] = input[IDX2R(r,c,input_d1)];
                        argmax[IDX2R(e,c,input_d1)] = r;
                    }
                }
            }
        }

        // Copy the gradients from the previous one, depending on the indexes that got
        // the maximums
        for (int e = 0; e < offset_size-1; ++e){
            for (int c = 0; c < n_filter; ++c){
                int r = argmax[IDX2R(e,c,input_d1)];
                output_gradient[IDX2R(r,c,input_d1)] = input_gradient[IDX2R(e,c,input_d1)];
            }
        }

        /*printf("MP INPUT GRADIENT\n");
        for (int e = 0; e < 50; ++e){
            for (int c = 0; c < 10; ++c){
                printf("%2.10f ", input_gradient[IDX2R(e,c,input_d1)]);
            }
            printf("\n");
        }*/

        delete[] max;
        delete[] argmax;
    }
};


template <typename Device, typename T>
class DownsampleSentenceMaxGradOp : public OpKernel {
	public:
		explicit DownsampleSentenceMaxGradOp(OpKernelConstruction* context) : OpKernel(context){}
        void Compute(OpKernelContext* context) override {
            const Tensor& input_tensor = context->input(0);
            const Tensor& offset_tensor = context->input(1);
            const Tensor& input_gradient_tensor = context->input(2);

            OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor.shape()),
                    errors::InvalidArgument("DownsampleSentenceMax expects a 2-D input matrix."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(offset_tensor.shape()),
                    errors::InvalidArgument("DownsampleSentenceMax expects a 1-D offset vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_gradient_tensor.shape()),
                    errors::InvalidArgument("DownsampleSentenceMax expects a 2-D offset vector."));

            // Grab the input tensor and the input offset tensor
            auto input = input_tensor.matrix<T>();
            //printf("INPUT MATRIX SIZE: %d %d\n", input_tensor.shape().dim_size(0), input_tensor.shape().dim_size(1));
            auto offset = offset_tensor.flat<int64>();
            //printf("OFFSET SIZE: %d\n", offset_tensor.shape().dim_size(0));
            auto gz = input_gradient_tensor.matrix<T>();
            //printf("GRADIENT MATRIX SIZE: %d %d\n", input_gradient_tensor.shape().dim_size(0), input_gradient_tensor.shape().dim_size(1));
            // Create an output tensor
            // The resulting number of rows will be equal to len(offset)-1
            auto new_tensor_shape = input_tensor.shape();
            new_tensor_shape.set_dim(0, offset.size() - 1);
            // Alloc memory for the output_tensor
            Tensor *output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
            auto gx = output_tensor->matrix<T>(); //gx is the output gradient and has the same shape as the input tensor
            //OP_REQUIRES(context, (input_tensor.shape().dim_size(0) == offset(offset.size()-1)),\
                    errors::InvalidArgument("Input tensor shape does not match len(offset)-1 ", input_tensor.shape().dim_size(0), " ", offset(offset.size()-1)));

            DownsampleSentenceMaxGradFunctor<Device, T>()(
                    context->eigen_device<Device>(),
                    static_cast<int>(input_tensor.shape().dim_size(0)),
                    static_cast<int>(input_tensor.shape().dim_size(1)),
                    static_cast<int>(offset_tensor.NumElements()),
                    input.data(),
                    offset.data(),
                    gz.data(),
                    gx.data());
        }
};

#define REGISTER_CPU(T)                                                           \
    REGISTER_KERNEL_BUILDER(                                                            \
            Name("DownsampleSentenceMaxGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
            DownsampleSentenceMaxGradOp<CPUDevice,T>)
REGISTER_CPU(float)

// Register the GPU kernels
#if NUANCE_CUDA
#define REGISTER_GPU(T)                                                                 \
    REGISTER_KERNEL_BUILDER(                                                            \
            Name("DownsampleSentenceMaxGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
            DownsampleSentenceMaxGradOp<GPUDevice, T>);

//REGISTER_GPU(int32);
REGISTER_GPU(float);
#endif  // NUANCE_CUDA

