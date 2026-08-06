
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <stdio.h>
#include <iostream>
#include <Windows.h>
#include <memoryapi.h>

// compile time constant for tile size
#define TWIDTH 32

struct fail_exc {
    std::string reason;
};

cudaError_t matrixMult(double* mR, const double* mA, const double* mB, uint32_t width);

__global__ void matrixMultK(double *mR, const double* mA, const double *mB, uint32_t width)
{
    // CUDA coordinates
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // matrix coordinates
    unsigned int x = bx * TWIDTH + tx;
    unsigned int y = by * TWIDTH + ty;

    // array index
    unsigned int i = y * width + x;

    // input tiles in shared memory
    __shared__ double tileA[TWIDTH][TWIDTH], tileB[TWIDTH][TWIDTH];

    // will accumulate computations and ultimately contain R[x,y].
    double res{};

    // each iteration calculates the contribution of two square input tiles
    // to the result
    for (int m = 0; m < width/TWIDTH; ++m) {
        // phase 1: parallel load of shared memory
        tileA[tx][ty] = mA[y * width + m * TWIDTH + tx];
        tileB[tx][ty] = mB[((m * TWIDTH) + ty)* width + x];
        __syncthreads();

        // phase 2: partial dot product up to the current input tiles
        for (int k = 0; k < TWIDTH; ++k)
            res += tileA[tx][k] * tileB[k][ty];

        // avoid starting new iteration before everyone is done
        __syncthreads();
    }
    
    mR[i] = res;
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

// Initializes the matrix with a certain value (sqrt(2)/2) on the diagonal.
// Assumes m is initialized to zero
void initialize_matrix(double* m, uint32_t width)
{
    double val = sqrt(2.0) / 2.0;
    for (uint32_t i = 0; i < width; ++i)
        m[i * width + i] = val;
}

int main()
{
    // We create square input matrices that can be perfectly cut into 32x32 tiles.
    // The easiest choice is to use powers of 2^k where k >= 5. This ensures we can
    // repeatedly divide by powers of 2 whenever it's useful to do so

    const int width = 1 << 9;
    double* mA{}, *mB{}; // input matrices 
    double* mR{}; // output matrix
    cudaError_t cudaStatus;

    try {
        // Allocate matrices
        mA = reinterpret_cast<double*>(host_alloc(width * width * sizeof(double)));
        mB = reinterpret_cast<double*>(host_alloc(width * width * sizeof(double)));
        mR = reinterpret_cast<double*>(host_alloc(width * width * sizeof(double)));

        // 
        initialize_matrix(mA, width);
        initialize_matrix(mB, width);

        // ...
        cudaStatus = matrixMult(mR, mA, mB, width);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addWithCuda failed!");
            return 1;
        }

    }
    catch (fail_exc e) {
        std::cerr << "Host error: " << e.reason << std::endl;
    }

    // cleanup
    if (mA)
        host_free(mA);
    if (mB)
        host_free(mB);
    if (mR)
        host_free(mR);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


}

// Helper function for using CUDA to perform R = A*B for matrices A, B, R.
cudaError_t matrixMult(double* mR, const double* mA, const double* mB, uint32_t width)
{
    double* dev_mA{}, * dev_mB{}, * dev_mR{};
    cudaError_t cudaStatus;

    dim3 dGrid(width / TWIDTH, width / TWIDTH, 1);
    dim3 dBlock(TWIDTH, TWIDTH, 1);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_mR, width * width * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mA, width * width * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mB, width * width * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_mA, mA, width * width * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_mB, mB, width * width * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    matrixMultK<<<dGrid, dBlock>>>(dev_mR, dev_mA, dev_mB, width);

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
    cudaStatus = cudaMemcpy(mR, dev_mR, width * width * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    std::cout << "Multiplication result at (0,0) = " << mR[0] << std::endl << std::endl;

Error:
    cudaFree(dev_mR);
    cudaFree(dev_mA);
    cudaFree(dev_mB);
    
    return cudaStatus;
}
