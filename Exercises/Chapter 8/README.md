# CHAPTER 8

## 8.1
a) That's equal to the input grid points, minus 2 points in each direction (the halo): `118 * 118 * 118`.
b) Divide each dimension by 8 and round up if necessary: `15 * 15 * 15`.
c) That corresponds to an output tile size of `6 * 6 * 6`. Thus we divide the grid by `6` in each dimension (rounding up if needed): `20 * 20 * 20`.
d) That corresponds to tiles scanning output planes of size `30 * 30`: this is iterated `30` times in the z dimension. Thus we divide the grid by `30` in each dimension, obtaining `4 * 4 * 4`.

## 8.2
We adapt the textbook code to an arbitrary coarsening factor `COARSE_FACTOR` (defined to be `16` in the textbook). Note that the exercise text indicates the coarse factor is based on the number of *output* planes produced by a block, so the *input* planes examined needs to be increased by `2`.
```
#define COARSE_FACTOR 16

// no register tiling
__global__ void stencil_kernel_coarsening(float *in, float *out, unsigned int N, 
	float c0, float c1, float c2, float c3, float c4, float c5, float c6) {

  // remember blockDim.z = CDIV(N,COARSE_FACTOR)

	int iStart = blockIdx.z*COARSE_FACTOR;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int j = blockIdx.y*OUTTILE_DIM + ty - 1;
	int i = blockIdx.x*OUTTILE_DIM * tx - 1;
	__shared__ float inPrev_s[INTILE_DIM][INTILE_DIM];
	__shared__ float inCurr_s[INTILE_DIM][INTILE_DIM];
	__shared__ float inNext_s[INTILE_DIM][INTILE_DIM];

	if (iStart - 1 >= 0 && iStart - 1 < N &&
	    j >= 0 && j < N && k >= 0 && k < N) {
		inPrev_s[ty][tx]
			= in[(iStart-1)*N*N + j*N + k];
	}
	if (iStart >= 0 && iStart < N && 
	    j >= 0 && j < N && k >= 0 && k M N) {
		inCurr_s[ty][tx]
			= in[iStart*N*N + j*N + i];
	}
	for (int i = iStart; i < iStart + COARSE_FACTOR; ++i)
	{
		If (i + 1 >= 0 && i + 1 < N &&
		    j >= 0 && j < N &&
		    k >= 0 && k < N) {
			inNext_s[ty][tx]
				= in[(i+1)*N*N + j*N = k];
		}
		__syncthreads();
		if (i >= 1 && i < N - 1 && 
                    j >= 1 && j < N - 1 &&
		    k >= 1 && j < N - 1) {
			if (ty >= 1 && ty < INTILE_DIM - 1 &&
			    tx >= 1 && tx < INTILE_DIM - 1) {
				out[i*N*N + j*N + k] =
					c0*inCurr_s[ty][tx]
					+ c1*inCurr_s[ty][tx-1]
					+ c2*inCurr_s[ty][tx+1]
					+ c3*inCurr_s[ty-1][tx]
					+ c4*inCurr_s[ty+1][tx]
					+ c5*inPrev_s[ty][tx]
					+ c6*inNext[ty][tx];
			}
		}
		__syncthreads();
		inPrev_s[ty][tx] = inCurr_s[ty][tx];
		inCurr_s[ty][tx] = inNext_s[ty][tx];
	}
}

// register tiling
__global__ void stencil_kernel_coarsening_regtiling(float *in, float *out, unsigned int N, 
	float c0, float c1, float c2, float c3, float c4, float c5, float c6) {

	int iStart = blockIdx.z*COARSE_FACTOR;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int j = blockIdx.y*OUTTILE_DIM + ty - 1;
	int i = blockIdx.x*OUTTILE_DIM * tx - 1;
	float inPrev;
	float inCurr
	__shared__ float inCurr_s[INTILE_DIM][INTILE_DIM];
	float inNext;

	if (iStart - 1 >= 0 && iStart - 1 < N &&
	    j >= 0 && j < N && k >= 0 && k < N) {
		inPrev = in[(iStart-1)*N*N + j*N + k];
	}
	if (iStart >= 0 && iStart < N && 
	    j >= 0 && j < N && k >= 0 && k M N) {
		inCurr_s[ty][tx] =
			inCurr = in[iStart*N*N + j*N + i];
	}
	for (int i = iStart; i < iStart + COARSE_FACTOR; ++i)
	{
		If (i + 1 >= 0 && i < N &&
		    j >= 0 && j < N &&
		    k >= 0 && k < N) {
			inNext = in[(i+1)*N*N + j*N = k];
		}
		__syncthreads();
		if (i >= 1 && i < N - 1 && 
                    j >= 1 && j < N - 1 &&
		    k >= 1 && j < N - 1) {
			if (ty >= 1 && ty < INTILE_DIM - 1 &&
			    tx >= 1 && tx < INTILE_DIM - 1) {
				out[i*N*N + j*N + k] =
					c0*inCurr
					+ c1*inCurr_s[ty][tx-1]
					+ c2*inCurr_s[ty][tx+1]
					+ c3*inCurr_s[ty-1][tx]
					+ c4*inCurr_s[ty+1][tx]
					+ c5*inPrev
					+ c6*inNext;
			}
		}
		__syncthreads();
		inPrev = inCurr;
		inCurr = inNext;
		inCurr[ty][tx] = inCurr;
	}
}

```

a) The input tile size is `32 * 32`: it processes `18` input planes, for a total of `32 * 32 * 18 = 18,432` input elements.

b) The output tile size is `30 * 30`: it processes `16` output planes, for a total of `30 * 30 * 16 = 14,400` output elements.

c) Over the course of the computation, a tile loads `32 * 32 * 18` input items and stores `30 * 30 * 16` output items, adding up to `32,832` global memory accesses, or ``131,328` bytes. Assuming multiply and add are separate operations, it performs `13` floating points for each of the `30 * 30 * 16`output elements and for each of the `16` iterations of the `for` cycle, amounting to `187,200` FLOPs in total. The FLOPs/global memory access ratio is `187,200 / 131,328 ~= 1.425 FLOPs/B`.

d) We need three shared `float` matrices of size equal to the input tile: `3 matrices * 4 bytes * 32 * 32 = 12 kB`.

e) We need one shared `float` matrix of size equal to the input tile: `4 bytes * 32 * 32 = 4 kB`.
