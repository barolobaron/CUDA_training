
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <Windows.h>
#include <memoryapi.h>

// define for block dimensions
#define BDIM2D 32
#define BDIM3D 8

// define for filter radius
#define FILTER_RADIUS 2

// define for input and output tile dimensions in 3D, assuming blocks are sized like input tiles
#define INTILE3D BDIM3D
#define OUTTILE3D (INTILE3D - 2*FILTER_RADIUS)

// define for input and output tile dimensions in 2D, assuming blocks are sized like output tiles
#define OUTTILE2D BDIM2D
#define INTILE2D (OUTTILE2D + 2*FILTER_RADIUS)

struct fail_exc {
    std::string reason;
};

// Exercise 7.8
// Basic 3D convolution
// In this kernel block dimensions and the filter radius are not hardcoded.
__global__ void convolution_3D_basicK(
    const float *N, const float *F, float *P, int r, int width, int height, int depth)
{
    // output coordinates
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outPln = blockIdx.z * blockDim.z + threadIdx.z;

    float Pvalue = 0.0f;

    // iterate on filter element coordinates
    for (int fPln = 0; fPln < 2 * r + 1; ++fPln)
        for (int fRow = 0; fRow < 2 * r + 1; ++fRow)
            for (int fCol = 0; fCol < 2 * r + 1; ++fCol) {
                int inPln = outPln - r + fPln;
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
        
                // do not access out of bounds elements in N
                if (inPln >= 0 && inPln < depth && inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    Pvalue += N[inPln * width * height + inRow * width + inCol]
                    * F[fPln * (2 * r + 1) * (2 * r + 1) + fRow * (2 * r + 1) + fCol];
    }
    // do not access out of bounds elements in P
    if (outPln >= 0 && outPln < depth && outRow >= 0 && outRow < height && outCol >= 0 && outCol < width)
        P[outPln * width * height + outRow * width + outCol] = Pvalue;
}

// declare a symbol in constant memory for filters -- filter radius needs to be known at compile time
__constant__ float cF[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// Exercise 7.9
// Basic 3D convolution with filter in constant memory.
// In this kernel block dimensions are not hardcoded.
__global__ void convolution_3D_const_memK(
    const float* N, float* P, int width, int height, int depth)
{
    // output coordinates
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outPln = blockIdx.z * blockDim.z + threadIdx.z;
    int fDim = 2 * FILTER_RADIUS + 1;

    float Pvalue = 0.0f;

    // iterate on filter element coordinates
    for (int fPln = 0; fPln < fDim; ++fPln)
        for (int fRow = 0; fRow < fDim; ++fRow)
            for (int fCol = 0; fCol < fDim; ++fCol) {
                int inPln = outPln - FILTER_RADIUS + fPln;
                int inRow = outRow - FILTER_RADIUS + fRow;
                int inCol = outCol - FILTER_RADIUS + fCol;

                // do not access out of bounds elements in N
                if (inPln >= 0 && inPln < depth && inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    Pvalue += N[inPln * width * height + inRow * width + inCol] * cF[fPln][fRow][fCol];
            }
    // do not access out of bounds elements in P
    if (outPln >= 0 && outPln < depth && outRow >= 0 && outRow < height && outCol >= 0 && outCol < width)
        P[outPln * width * height + outRow * width + outCol] = Pvalue;
}

// Exercise 7.10
// 3D tiled convolution, with input tile dimensions == block dimensions
__global__ void convolution_tiled_3D_const_memK(const float* N, float* P, int width, int height, int depth)
{
    int fDim = 2 * FILTER_RADIUS + 1;

    // thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // coordinates associated with this thread
    // (for input coordinates: coordinate in N -- may lie out of N, will read as zero)
    // (for output coordinates: coordinate in P -- ignored if it lies out of P)
    int col = blockIdx.x * OUTTILE3D + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUTTILE3D + threadIdx.y - FILTER_RADIUS;
    int pln = blockIdx.z * OUTTILE3D + threadIdx.z - FILTER_RADIUS;

    __shared__ float N_s[INTILE3D][INTILE3D][INTILE3D];

    // out of bounds elements of N read as zero
    if (pln >= 0 && pln < depth && row >= 0 && row < height && col >= 0 && col < width)
        N_s[tz][ty][tx] = N[pln * width * height + row * width + col];
    else
        N_s[tz][ty][tx] = 0.0f;
    __syncthreads();

    // top-left-front corner of coordinates in N_s
    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;
    int tilePln = tz - FILTER_RADIUS;

    // do not access P out of bounds
    if (pln >= 0 && pln < depth && row >= 0 && row < height && col >= 0 && col < width) {
        // do not access P outside the output tile bounds
        if (tileCol >= 0 && tileCol < OUTTILE3D &&
            tileRow >= 0 && tileRow < OUTTILE3D &&
            tilePln >= 0 && tilePln < OUTTILE3D) {

            float Pvalue = 0.0f;
            for (int fPln = 0; fPln < fDim; ++fPln)
                for (int fRow = 0; fRow < fDim; ++fRow)
                    for (int fCol = 0; fCol < fDim; ++fCol)
                        Pvalue += N_s[tilePln + fPln][tileRow + fRow][tileCol + fCol] * cF[fPln][fRow][fCol];

            P[pln * width * height + row * width + col] = Pvalue;
        }
    }
}


// like cF, allocate constant memory for filters, but 2D
__constant__ float cF2D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// Exercise 7.11
// 2D tiled convolution, with output tile dimensions == block dimensions
__global__ void convolution_tiled_alt_2D_const_memK(const float* N, float* P, int width, int height)
{
    int fDim = 2 * FILTER_RADIUS + 1;

    // thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // output coordinates correspond to block+thread coordinates
    int outcol = blockIdx.x * blockDim.x + tx;
    int outrow = blockIdx.y * blockDim.y + ty;

    __shared__ float N_s[INTILE2D][INTILE2D];

    // The thread block is sized against output tiles, so a thread may need to load
    // more than a value from the larger input tile.
    // The indices h and k are used to iterate through the input tile
    for (int h = 0; h < (INTILE2D + OUTTILE2D -1)/OUTTILE2D; ++h)
        for (int k = 0; k < (INTILE2D + OUTTILE2D - 1) / OUTTILE2D; ++k) {
            // base input coordinates are shifted by (- FILTER_RADIUS)
            int bincol = outcol - FILTER_RADIUS;
            int binrow = outrow - FILTER_RADIUS;
            // do not write elements outside N_s
            if (h * OUTTILE2D + tx < INTILE2D && k * OUTTILE2D + ty < INTILE2D) {
                // read elements outside N as zero
                if (bincol + h * OUTTILE2D >= 0 && bincol + h * OUTTILE2D < width &&
                    binrow + k * OUTTILE2D >= 0 && binrow + k * OUTTILE2D < height) {
                    N_s[k * OUTTILE2D + ty][h * OUTTILE2D + tx] =
                        N[(binrow + k * OUTTILE2D)*width + bincol + h * OUTTILE2D];
                }
                else {
                    N_s[k * OUTTILE2D + ty][h * OUTTILE2D + tx] = 0.0f;
                }
            }
        }

    __syncthreads();

    float Pvalue = 0.0f;

    // do not access P out of bounds
    if (outrow < height && outcol < width) {
        for (int fRow = 0; fRow < fDim; ++fRow)
            for (int fCol = 0; fCol < fDim; ++fCol)
                Pvalue += N_s[ty + fRow][tx + fCol] * cF2D[fRow][fCol];

        P[outrow * width + outcol] = Pvalue;
    }
}

cudaError_t convolution_3D_basic(float* N, float* F, float* P, int r, int width, int height, int depth);
cudaError_t convolution_3D_const_mem(float* N, float* F, float* P, int width, int height, int depth);
cudaError_t convolution_tiled_3D_const_mem(float* N, float *F, float* P, int width, int height, int depth);
cudaError_t convolution_tiled_alt_2D_const_mem(float* N, float *F, float* P, int width, int height);

// Allocated region is guaranteed to be initialized to zero.
LPVOID host_alloc(uint32_t len) {
    LPVOID ptr;
    if ((ptr = VirtualAlloc(NULL, len, MEM_COMMIT, PAGE_READWRITE)) == NULL)
        throw fail_exc{ std::string("VirtualAlloc failed.") };
    return ptr;
}

BOOL host_free(LPVOID addr) {
    return VirtualFree(addr, 0, MEM_RELEASE);
}

// Initializes float data (a float array)
void initialize_float_data(float* v, int len)
{
    for (int i = 0; i < len; ++i)
        v[i] = 1.0;
}

float F2D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {0.01f,  0.02f,   0.05f,   0.02f,   0.01f},
    {0.02f,  0.04f,   0.1f,    0.04f,   0.02f},
    {0.05f,  0.1f,    0.25f,   0.1f,    0.05f},
    {0.02f,  0.04f,   0.1f,    0.04f,   0.02f},
    {0.01f,  0.02f,   0.05f,   0.02f,   0.01f}
};

float F3D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {
        {0.0005f, 0.001f,  0.0025f,   0.001f,   0.0005f},
        {0.001f,  0.002f,  0.005f,    0.002f,   0.001f},
        {0.0025f, 0.005f,  0.0125f,   0.005f,   0.0025f},
        {0.001f,  0.002f,  0.005f,    0.002f,   0.001f},
        {0.0005f, 0.001f,  0.0025f,   0.001f,   0.0005f}
    },
    {
        {0.001f, 0.002f,  0.005f,   0.002f,   0.001f},
        {0.002f, 0.004f,  0.01f,    0.004f,   0.002f},
        {0.005f, 0.01f,   0.025f,   0.01f,    0.005f},
        {0.002f, 0.004f,  0.05f,    0.004f,   0.002f},
        {0.001f, 0.002f,  0.025f,   0.002f,   0.001f}
    },
    {
        {0.005f, 0.01f,   0.025f,   0.01f,   0.005f},
        {0.01f,  0.02f,   0.05f,    0.02f,   0.01f},
        {0.025f, 0.05f,   0.125f,   0.05f,   0.025f},
        {0.01f,  0.02f,   0.05f,    0.02f,   0.01f},
        {0.005f, 0.01f,   0.025f,   0.01f,   0.005f}
    },
    {
        {0.001f, 0.002f,  0.005f,   0.002f,   0.001f},
        {0.002f, 0.004f,  0.01f,    0.004f,   0.002f},
        {0.005f, 0.01f,   0.025f,   0.01f,    0.005f},
        {0.002f, 0.004f,  0.05f,    0.004f,   0.002f},
        {0.001f, 0.002f,  0.025f,   0.002f,   0.001f}
    },
    {
        {0.0005f, 0.001f,  0.0025f,   0.001f,   0.0005f},
        {0.001f,  0.002f,  0.005f,    0.002f,   0.001f},
        {0.0025f, 0.005f,  0.0125f,   0.005f,   0.0025f},
        {0.001f,  0.002f,  0.005f,    0.002f,   0.001f},
        {0.0005f, 0.001f,  0.0025f,   0.001f,   0.0005f}
    }
};

int main()
{
    float* N2D{}, * N3D{}; // input vectors
    float* P2D{}, * P3D{}; // output vectors
    // filter arrays as pointers for convenience
    float* pF2D = reinterpret_cast<float*>(F2D);
    float* pF3D = reinterpret_cast<float*>(F3D);

    cudaError_t cudaStatus;
    int failure = 0; // not failed by default
    int len2D = 1024;
    int len3D = 512;
    int sz2D = len2D * len2D;
    int sz3D = len3D * len3D * len3D;

    try {
        // Allocate vectors
        N2D = reinterpret_cast<float*>(host_alloc(sz2D * sizeof(float)));
        N3D = reinterpret_cast<float*>(host_alloc(sz3D * sizeof(float)));
        P2D = reinterpret_cast<float*>(host_alloc(sz2D * sizeof(float)));
        P3D = reinterpret_cast<float*>(host_alloc(sz3D * sizeof(float)));

        // put some values in the input
        initialize_float_data(N2D, sz3D);
        initialize_float_data(N3D, sz3D);

        // run the kernels
        // Test 1
        cudaStatus = convolution_3D_basic(N3D, pF3D, P3D, FILTER_RADIUS, len3D, len3D, len3D);

        if (cudaStatus != cudaSuccess) {
            throw fail_exc{ std::string("convolution_3D_basic failed!") };
        }

        // Test 2
        cudaStatus = convolution_3D_const_mem(N3D, pF3D, P3D, len3D, len3D, len3D);

        if (cudaStatus != cudaSuccess) {
            throw fail_exc{ std::string("convolution_3D_const_mem failed!") };
        }

        // Test 3
        cudaStatus = convolution_tiled_3D_const_mem(N3D, pF3D, P3D, len3D, len3D, len3D);

        if (cudaStatus != cudaSuccess) {
            throw fail_exc{ std::string("convolution_tiled_3D_const_mem failed!") };
        }

        // Test 4
        cudaStatus = convolution_tiled_alt_2D_const_mem(N2D, pF2D, P2D, len2D, len2D);

        if (cudaStatus != cudaSuccess) {
            throw fail_exc{ std::string("convolution_tiled_alt_2D_const_mem failed!") };
        }
    }
    catch (fail_exc e) {
        std::cerr << e.reason << std::endl;
        failure = 1;
        goto cleanup;
    }

    /* TODO output
    std::cout << "Output vector =" << std::endl;
    for (int i = 0; i < len; ++i)
        std::cout << vR[i] << " ";

    std::cout << std::endl << std::endl;
    */

cleanup:
    if (N2D)
        host_free(N2D);
    if (N3D)
        host_free(N3D);
    if (P2D)
        host_free(P2D);
    if (P3D)
        host_free(P3D);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return failure;

}

// Helper function for using CUDA to perform R = A*B for matrices A, B, R.
cudaError_t convolution_3D_basic(float *N, float *F, float *P, int r, int width, int height, int depth)
{
    float* dev_N{}, * dev_F{}, * dev_P{};
    int sizeData = sizeof(float) * width * height * depth;
    int sizeF = sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);
    cudaError_t cudaStatus;

    int gridx = (width + 15) / 16;
    int gridy = (height + 7) / 8;
    int gridz = (depth + 7) / 8;

    dim3 dGrid(gridx, gridy, gridz);
    dim3 dBlock(16, 8, 8);


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_N, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_F, sizeF);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_P, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_N, N, sizeData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_F, F, sizeF, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    convolution_3D_basicK <<< dGrid, dBlock >>> (dev_N, dev_F, dev_P, r, width, height, depth);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(P, dev_P, sizeData, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_N);
    cudaFree(dev_F);
    cudaFree(dev_P);

    return cudaStatus;
}

cudaError_t convolution_3D_const_mem(float* N, float* F, float* P, int width, int height, int depth) {
    float* dev_N{}, * dev_P{};
    int sizeData = sizeof(float) * width * height * depth;
    int sizeF = sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);
    cudaError_t cudaStatus;

    int gridx = (width + BDIM3D - 1) / BDIM3D;
    int gridy = (height + BDIM3D - 1) / BDIM3D;
    int gridz = (depth + BDIM3D - 1) / BDIM3D;

    dim3 dGrid(gridx, gridy, gridz);
    dim3 dBlock(BDIM3D, BDIM3D, BDIM3D);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_N, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_P, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_N, N, sizeData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyToSymbol(cF, F, sizeF);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    convolution_3D_const_memK <<< dGrid, dBlock >>> (dev_N, dev_P, width, height, depth);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(P, dev_P, sizeData, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_N);
    cudaFree(dev_P);

    return cudaStatus;

}

cudaError_t convolution_tiled_3D_const_mem(float* N, float* F, float* P, int width, int height, int depth) {
    float* dev_N{}, * dev_P{};
    int sizeData = sizeof(float) * width * height * depth;
    int sizeF = sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);
    cudaError_t cudaStatus;

    int gridx = (width + BDIM3D - 1) / BDIM3D;
    int gridy = (height + BDIM3D - 1) / BDIM3D;
    int gridz = (depth + BDIM3D - 1) / BDIM3D;

    dim3 dGrid(gridx, gridy, gridz);
    dim3 dBlock(BDIM3D, BDIM3D, BDIM3D);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_N, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_P, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_N, N, sizeData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyToSymbol(cF, F, sizeF);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    convolution_tiled_3D_const_memK <<< dGrid, dBlock >>> (dev_N, dev_P, width, height, depth);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(P, dev_P, sizeData, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_N);
    cudaFree(dev_P);

    return cudaStatus;


}

cudaError_t convolution_tiled_alt_2D_const_mem(float* N, float* F, float* P, int width, int height) {
    float* dev_N{}, * dev_P{};
    int sizeData = sizeof(float) * width * height;
    int sizeF = sizeof(float) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);
    cudaError_t cudaStatus;

    int gridx = (width + BDIM2D - 1) / BDIM2D;
    int gridy = (height + BDIM2D - 1) / BDIM2D;
    int gridz = 1;

    dim3 dGrid(gridx, gridy, gridz);
    dim3 dBlock(BDIM2D, BDIM2D, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_N, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_P, sizeData);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_N, N, sizeData, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpyToSymbol(cF2D, F, sizeF);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    convolution_tiled_alt_2D_const_memK << < dGrid, dBlock >> > (dev_N, dev_P, width, height);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(P, dev_P, sizeData, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_N);
    cudaFree(dev_P);

    return cudaStatus;

}
