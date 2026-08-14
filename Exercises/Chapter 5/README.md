# CHAPTER 5

## 5.1
No, because every thread computes its output using different elements of the input.

## 5.2
With 2x2 tiling, each tile performs 2 * 4 * (2 * 8)/4 = 32 loads.
With 4x4 tiling, each tile performs 2 * 16 * (4 * 8)/16 = 64 loads.
Without tiling, each thread performs 2 * 8 = 16 loads.

Therefore the relative bandwidth usage of the tiled algorithm compared to the untiled one is:
2x2: (32 loads / 4 threads per tile)/16 = 50%
4x4: (64 loads / 16 threads per tile)/16 = 25%

## 5.3
Without the first `__syncthreads()`, a thread could read an old (or uninitialized) value from `Mds` or `Nds`.
Without the second `__syncthreads()`, a thread may overwrite an element of `Mds` or `Nds` before all the other threads have been able to use it.
Both statements are therefore required to ensure correctness.

## 5.4
Shared memory allows to share values between the threads in the same block. Updates made to shared memory are immediately visible to the threads in the same block, whereas updates made to registers are only visible through the use of global memory.

Furthermore, shared memory can contain data structures of a size that will only be known at runtime, whereas registers are statically allocated.

## 5.5
31/32 = ~96.9%

## 5.6
512,000

## 5.7
1000

## 5.8
a: N, b: ceiling(N/T)

## 5.9
a) memory-bound, because 200/36 > 100/(4 * 7)
b) compute-bound, because 300/36 < 250/(4 * 7)

## 5.10
Assuming `BLOCK_WIDTH` and `BLOCK_SIZE` are aliases.
a) `BLOCK_WIDTH` must divide `A_width` and `A_height` exactly. Furthermore, there's no synchronization between lines 10--11.

This implies `BLOCK_WIDTH` = 1 (though the second requirement alone could in theory work for values up to 4 because 4x4 = 16 < 32 threads/warp, but as we know assuming the threads in the same warp execute the same instruction is a bad idea.

b) correct line 2 by rounding up the division (i.e. A_width/blockDim.x becomes A_width/ceil((float) blockDim.x), and similarly for A_height/blockDim.y.

Wrap the lines 10-11 in a conditional for limits checking.

Add __syncthreads() between lines 10--11.

## 5.11
The kernel is run with 8 blocks of 128 threads each, yielding a total of 1024 threads..

a) 1024
b) 1024
c) 8
d) 8
e) 516 bytes
f) 6 floating point operations (using `FMA` = multiply and add), 6 load/store operations x 4 bytes each, for a total of 0.25 floating point operations per byte.

## 5.12
a) will run 2048 threads max using all 32 blocks.
27 x 2048 registers < 64K and 8 < 96, Full!

b) will run 2048 threads (the maximum) using 8 blocks:
31 x 2048 < 64K, and 8 < 96, full! (even though 8 blocks < 32 blocks max)

(if however the shared memory in the kernel is expressed per block and not per SM, we have:
a) 4 x 32 > 96 (limited by shared memory)
b) 8 x 8 < 96 (full!))

