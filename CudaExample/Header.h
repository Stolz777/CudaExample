#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cuda.h>
#include <uchar.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__host__ std::string isFolder(std::string folder);
__host__ void CallCudaKernel(uchar* d_data, size_t rows, size_t columns, int threadsPerBlock);
__global__ void InvertMultiImage(uchar* d_data, int size);