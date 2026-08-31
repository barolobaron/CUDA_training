# CHAPTER 7

## 7.1
Assuming `P' is the output vector, i.e. `y' in the picture.

The input elements x[-1] and x[-2] do not exist so they are ignored/treated as zero. We get:
```
y[0] = x[0]*f[0] + x[1]*f[1] + x[2]*f[2]
     = 5*8 + 3*2 + 1*5
     = 51
```

## 7.2
For `N = (4,1,3,2,3)` and `F = (2,1,4)`
```
P[0] = 4*1 + 1*4 = 8
P[1] = 4*2 + 1*1 + 3*4 = 21
P[2] = 1*2 + 3*1 + 2*4 = 13
P[3] = 3*2 + 2*1 + 3*4 = 20
P[4] = 2*2 + 4*1 = 8

P = (8, 21, 13, 20, 8)
```

## 7.3
a) identity
b) successor/shift left
c) predecessor/shift right
d) average of right neighbour and opposite of left neighbour
e) average of the value and its two neighbours

## 7.4
a) `M - 1`
b) `M * N`
c) `M * N - (M^2 - 1)/4`

## 7.5
a) `2*N*(M - 1) + (M - 1)^2`
b) `(M * N)^2`
c) This is a very boring arithmetical exercise. My notes give the result
`((3*M^2 - 4*M + 1)/8)*(M*N - (M^2 - 1)/4))`

## 7.6
a) `(M_1 - 1)*(M_2 - 1) + N_1*(M_1 - 1) + N_2*(M_2 - 1)
b) `M_1*M_2*N_1*N_2
c) Another boring arithmetical exercise. I can't find the solution in muy notes, but I don't think this is formative for an graduate engineer. We are trained in mathematics and we can use it when we need to, but here it's pointless.

## 7.7
a) `ceiling(N/T)^2`
b) `(T + M - 1)^2`
c) `4*(T + M - 1)^2`
d) `4*T^2`

## 7.8
See function `convolution_3D_basicK` in `Convolution\kernel.cu`

## 7.9
See function `convolution_3D_const_memK` in `Convolution\kernel.cu`

## 7.10
See function `convolution_tiled_3D_const_memK` in `Convolution\kernel.cu`

## 7.11
See function `convolution_tiled_alt_2D_const_memK` in `Convolution\kernel.cu`
