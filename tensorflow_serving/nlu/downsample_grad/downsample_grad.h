#ifndef DOWNSAMPLEGRAD_H_
#define DOWNSAMPLEGRAD_H_

#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

template <typename Device, typename T>
struct DownsampleSentenceMaxGradFunctor {
    void operator()(const Device &d, const int input_d0, const int input_d1, const int offset_size,
            const T *input, const int64 *offset, const T *input_gradient, T *output_gradient);
};

#endif // DOWNSAMPLEGRAD_H_
