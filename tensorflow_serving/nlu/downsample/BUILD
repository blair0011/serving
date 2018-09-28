package(
    default_visibility = [
        "//tensorflow_serving:internal",
    ],
    features = ["-layering_check"],
)

cc_library(
    name = "downsample.so",
    visibility = ["//visibility:public"],
    srcs = [
            "downsample.cpp",
            "downsample.h"
           ] ,
    copts = ["-std=c++11",
               "-shared",   
               "-fPIC",
               "-D_GLIBCXX_USE_CXX11_ABI=0"],
    deps = ["@org_tensorflow//tensorflow/core:framework_headers_lib",
            "@org_tensorflow//tensorflow/core/util/ctc",
            "@org_tensorflow//tensorflow/core/kernels:reader_ops",
            "@org_tensorflow//third_party/eigen3",
    ],
    alwayslink = 1,
)

