# CHAPTER 6

## 6.1
a) yes
b) N/A
c) yes
d) no
e) N/A
f) N/A
g) yes
h) N/A
i) no

## 6.2
The function `matrixMultKCT` in `MatrixMult\kernel.cu` implements square matrix multiplication
when the second input matrix `mB` is provided in column-major order, using corner-turning to
transfer tiles from `mB` into shared memory.

## 6.3
For coalesced access, we need the threads of a warp to access entire DRAM bursts.
Assuming a matrix of floats (4 bytes each) and warps of size 32, 
and noting that `BLOCK_SIZE` is up to 32 (it can't be greater than that, otherwise
square blocks would contain more than 1024 threads, which isn't allowed),
we note that `BLOCK_SIZE` should divide 32 exactly, and `4*BLOCK_SIZE` should
be a multiple of the size of a DRAM burst: for instance, for 64 byte DRAM bursts,
we can allow a `BLOCK_SIZE` of 16 or 32.

## 6.4
See `VectorAddK` in `VectorAdd\kernel.cu`.

## 6.5
The text of the exercise is ambiguous because the correspondence between thread indices in a
block and thread indices in a warp depends on the dimension of blocks. For simplicity, assume 
that blocks are 1-dimensional: then we know thread `0` in a warp corresponds to a thread 
index `k*32` in a block, with `k` a positive integer.

Then for a given warp, we have that thread indices (in the block) are in the interval
`k*32 ... k*32 + 31`. For the (0-based) `i`-th thread in a warp, the thread index in the block
is `x = k*32 + i`.

To know the bank accessed by the `i`-th thread in a warp, we need to take the integer division
of `x * stride * 4` by `128` (`4` is the size of a float, `128` is the size of a bank,
both in bytes). This is `((k * 32 + i)*stride*4) // 128 == (128 * k * stride + 4 * i * stride) // 128`
whioh further simplifies to `k * stride + (4 * i * stride) // 128`. This value, modulo 32 (the
number of banks), is the bank accessed by the `i`-th thread in a warp.

Therefore the bank accessed by the `0`-th thread in a warp depends on `k` and on `stride` (e.g.
if `stride == 31` and `k == 0`, thread `0` in the warp will access bank `0`; but if `k == 1`,
it will access bank `31`. In general, bank accesses for a given `k` are shifted by `k * stride`.

To simplify calculations, we will assume `k == 0`, and point out that when `k` is another
value bank accesses are shifted by `k * stride`.

a. banks 0-31 (no conflict)

b. banks 0-30 (bank 0 conflict for `i in { 0, 1 }`)

c. banks 0-23 (for `(i % 4) in { 0 , 1 }` we get a conflict on bank `3 * (i // 4)`

d. banks 0-15 (each bank `b` is accessed by threads `2b` and `2b + 1` in the warp

e. banks 0-11 
```
12 * 4 == 48` and `lcm(48, 128) == 384`, so we are squeezing every `384/48 == 8`
threads into `384/128 == 3` banks:

bank 0 <- i in { 0, 1, 2 }
bank 1 <- i in { 3, 4, 5 }
bank 2 <- i in { 6, 7 }
and then cyclically
bank 3 <- i in { 8, 9, 10 }
bank 4 <- i in { 11, 12, 13 }`
bank 5 <- i in { 14, 15 }
etc.
```

f. banks 0-7 (each bank `b` is accessed by threads `4b + m` in the warp, where `m in { 0, 1, 2, 3 }`

g. banks 0-6 
```
7 * 4 == 28` and lcm(28, 128) == 896, so we are squeezing every 896/28 == 32
threads in 896/128 == 7 banks... this is not of great help!
There is no cycle shorter than 32, so we have to check each thread individually.
We get:
bank 0 <- i in { 0 ... 4 }
bank 1 <- i in { 5 ... 9 }
bank 2 <- i in { 10 ... 13 }
bank 3 <- i in { 14 ... 18 }
bank 4 <- i in { 19 ... 22 }
bank 5 <- i in { 23 ... 27 }
bank 6 <- i in { 28 ... 31 }
```
