# CHAPTER 9

## 9.1
If one atomic operation takes `100 * 10^-9 s`, then the throughput is `(100 * 10^-9)^-1 == 10^7 op/s`.

## 9.2
We calculate:
```
(0.9 * 4 * 10^-9 + 0.1 * 100 * 10^-9)^-1
== (13.6 * 10^-9)^-1
~= 0.074 * 10^9
== 7.4 * 10^7 op/s
```
## 9.3
We just need to multiply the global memory throughput by `5 FLOPs/memopry access`, obtaining `5 * 10^7 FLOPs/s`.

## 9.4
The amortized time for each memory access is `1.1 * 10^-9 s`.
Then the memory throughput is `(1.1 * 10^-9)^-1 ~= 0.91 * 10^9 mem access/s`.
At `5 FLOPs/mem access` we get `5 * (0.91 * 10^9) = 4.55 * 10^9 FLOPs/s`.

## 9.5
a) same as number of input elements: `524,288`
b) `#bins * #blocks == 128 * 512 == 65,536`
c) `#bins * #blocks == 128 * 128 == 16,384` 
