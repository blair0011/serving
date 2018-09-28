#ifndef DOWNSAMPLE_H_
#define DOWNSAMPLE_H_

#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

template <typename Device, typename T>
struct DownsampleSentenceMaxFunctor {
	void operator()(const Device &d, const int input_d0, const int input_d1, const int offset_size,
                    const T *input, const int64 *offset, T *output);
};

#endif // DOWNSAMPLE_H_
