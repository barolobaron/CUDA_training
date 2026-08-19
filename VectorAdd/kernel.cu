
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <Windows.h>
#include <memoryapi.h>

// compile time constant for tile size
#define TWIDTH 32

struct fail_exc {
    std::string reason;
};

cudaError_t vectorAdd(double* vR, const double* vA, const double* vB, uint32_t len);

__global__ void vectorAddK(double* vR, const double* vA, const double* vB, uint32_t len)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i * 4 + 3 < len) {
        // there's at least 4 items left
        double4 x4 = reinterpret_cast<const double4*>(vA)[i];
        double4 y4 = reinterpret_cast<const double4*>(vB)[i];
        double4 z4;
        z4.x = x4.x + y4.x;
        z4.y = x4.y + y4.y;
        z4.z = x4.z + y4.z;
        z4.w = x4.w + y4.w;
        reinterpret_cast<double4*>(vR)[i] = z4;
    }
    else {
        switch (len - i * 4) {
        case 3:
            vR[i * 4 + 2] = vA[i * 4 + 2] + vB[i * 4 + 2];
        case 2:
            vR[i * 4 + 1] = vA[i * 4 + 1] + vB[i * 4 + 1];
        default: // 1
            vR[i * 4] = vA[i * 4] + vB[i * 4];
        }
    }
}

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

// Initializes the vector with items equal to theit index
void initialize_vector(double* v, uint32_t len)
{
    for (uint32_t i = 0; i < len; ++i)
        v[i] = (double) i;
}

int main()
{
    const uint32_t len = 1 << 16; // 2^16
    double* vA{}, * vB{}; // input vectors
    double* vR{}; // output vectors
    cudaError_t cudaStatus;
    int failure = 0; // not failed by default

    try {
        // Allocate vectors
        vA = reinterpret_cast<double*>(host_alloc(len * sizeof(double)));
        vB = reinterpret_cast<double*>(host_alloc(len * sizeof(double)));
        vR = reinterpret_cast<double*>(host_alloc(len * sizeof(double)));

        // put some values in the input
        initialize_vector(vA, len);
        initialize_vector(vB, len);

        // run the addition
        cudaStatus = vectorAdd(vR, vA, vB, len);

        if (cudaStatus != cudaSuccess) {
            throw fail_exc{ std::string("vectorAdd failed!") };
        }

    }
    catch (fail_exc e) {
        std::cerr << e.reason << std::endl;
        failure = 1;
        goto cleanup;
    }

    std::cout << "Output vector =" << std::endl;
    for (int i = 0; i < len; ++i)
        std::cout << vR[i] << " ";

    std::cout << std::endl << std::endl;

cleanup:
    if (vA)
        host_free(vA);
    if (vB)
        host_free(vB);
    if (vR)
        host_free(vR);

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
cudaError_t vectorAdd(double* vR, const double* vA, const double* vB, uint32_t len)
{
    double* dev_vA{}, * dev_vB{}, * dev_vR{};
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_vR, len * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vA, len * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_vB, len * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_vA, vA, len * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_vB, vB, len * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    vectorAddK <<<64, 1024>>> (dev_vR, dev_vA, dev_vB, len);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(vR, dev_vR, len * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_vR);
    cudaFree(dev_vA);
    cudaFree(dev_vB);

    return cudaStatus;
}
