# CHAPTER 4

## 4.1
The kernel is called on 8 blocks of 128 threads. Therefore:
a) 4 warps/block
b) 32 warps in the grid
c) i: 24 (= 3 * 8), ii: 16 (= 2 * 8), iii: 100%, iv: 25%, v: 75%
d) i: 36 (all), ii: 36 (all), iii: 50%
e) i: 3, ii: 2

## 4.2
2048

## 4.3
1 (threads: 1984--2015)

## 4.4
1 - (2.0 + 2.3 + 3.0 + 2.8 + 2.4 + 1.9 + 2.6 + 2.9)/(3*8) = ~ 17.1%

## 4.5
No. We shouldn't rely on the assunmption that all the threads in a warp are executing the same instruction, even if it worked for all current architectures.

## 4.6
C allows 1536 threads with 3 blocks, allowing the SM to run at full capacity. All the other options will not run at full capacity.

## 4.7
All are possible:
a) 50%, b) 50%, c) 50%, d) 100%, e) 100%

## 4.8
a) yes, b) limited by block slots, c) limited by registers

## 4.9
That cannot work because the kernel requires 1024 threads/block and the device only supports 512.
