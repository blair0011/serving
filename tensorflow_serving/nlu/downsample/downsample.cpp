
#define  EIGEN_USE_THREADS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "downsample.h"

#define IDX2R(i,j,ld) (((i)*(ld))+(j))

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("DownsampleSentenceMax")
    .Attr("T: realnumbertype")
    .Input("input_matrix: T")
    .Input("offset: int64")
    .Output("output: T")
    .Doc(R"doc(
    Computes the max pool operation over sequences delimited by the offset vector.
    output: A Tensor.
    )doc");

// CPU specialization of actual computation
template <typename T>
struct DownsampleSentenceMaxFunctor<CPUDevice, T>{
	void operator()(const CPUDevice &d, const int input_d0, const int input_d1, const int offset_size,
            const T *input, const int64 *offset, T *output){
        // input: Matrix with 2 dimentions of size [max_seq_len][n_filters]
        int n_filter = input_d1;
        // The output matrix will have the same shape as the input matrix, except that at
        // the 1st dimension will have as many elements as the number of maximums to
        // compute, which depend on the number of sequences we actually have: output[offset_size-1][n_filters]
        int output_d0 = offset_size - 1;
        int output_d1 = input_d1;

        for (int e = 0; e < output_d0; ++e){
            for (int c = 0; c < output_d1; ++c){
                output[IDX2R(e,c,output_d1)] = 0;
            }
        }

        for(int e = 0; e < offset_size-1; ++e){
            int start = offset[e];
            int end = offset[e+1];
            if (end == start)
                continue;
            for (int c = 0;  c < n_filter; ++c){
                output[IDX2R(e,c,output_d1)] = input[IDX2R(start,c,input_d1)];
            }
            for (int r = start+1; r < end; ++r){
                for (int c = 0; c < n_filter; ++c){
                    if (input[IDX2R(r,c,input_d1)] > output[IDX2R(e,c,output_d1)]) {
                        output[IDX2R(e,c,output_d1)] = input[IDX2R(r,c,input_d1)];
                    }
                }
            }
        }
	}
};

// OpKernel definition
// template parameter T is the datatype of tensors.
template <typename Device, typename T>
class DownsampleSentenceMaxOp : public OpKernel {
	public:
		explicit DownsampleSentenceMaxOp(OpKernelConstruction* context) : OpKernel(context){}
        void Compute(OpKernelContext* context) override {
            const Tensor& input_tensor = context->input(0);
            const Tensor& offset_tensor = context->input(1);

            OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor.shape()),
                    errors::InvalidArgument("DownsampleSentenceMax expects a 2-D input matrix."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(offset_tensor.shape()),
                    errors::InvalidArgument("DownsampleSentenceMax expects a 1-D offset vector."));

            // Grab the input tensor and the input offset tensor
            auto input = input_tensor.matrix<T>();
            auto offset = offset_tensor.flat<int64>();
            // Create an output tensor
            // The resulting number of rows will be equal to len(offset)-1
            auto new_tensor_shape = input_tensor.shape();
            new_tensor_shape.set_dim(0, offset.size() - 1);
            Tensor *output_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, new_tensor_shape, &output_tensor));
            auto output = output_tensor->template matrix<T>();

            int n_filter = new_tensor_shape.dim_size(1);

            /*OP_REQUIRES(context, (input_tensor.shape().dim_size(0) == offset(offset.size()-1)),
                    errors::InvalidArgument("Input tensor shape does not match len(offset)-1 ", input_tensor.shape().dim_size(0), " ", offset(offset.size()-1)));*/

            DownsampleSentenceMaxFunctor<Device, T>()(
                    context->eigen_device<Device>(),
                    static_cast<int>(input_tensor.shape().dim_size(0)),
                    static_cast<int>(input_tensor.shape().dim_size(1)),
                    static_cast<int>(offset_tensor.NumElements()),
                    input.data(),
                    offset.data(),
                    output.data());
        }
};

// Register the CPU kernels
#define REGISTER_CPU(T)                                                                 \
    REGISTER_KERNEL_BUILDER(                                                            \
            Name("DownsampleSentenceMax").Device(DEVICE_CPU).TypeConstraint<T>("T"),    \
            DownsampleSentenceMaxOp<CPUDevice, T>);
//REGISTER_CPU(int32);
REGISTER_CPU(float);

// Register the GPU kernels
#if NUANCE_CUDA
#define REGISTER_GPU(T)                                                                 \
    REGISTER_KERNEL_BUILDER(                                                            \
            Name("DownsampleSentenceMax").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
            DownsampleSentenceMaxOp<GPUDevice, T>);

//REGISTER_GPU(int32);
REGISTER_GPU(float);
#endif  // NUANCE_CUDA

//TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL)

