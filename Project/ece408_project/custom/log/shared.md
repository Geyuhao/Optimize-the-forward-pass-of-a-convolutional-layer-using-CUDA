# TILE_WIDTH = 16

## no modification
Conv-GPU==
Layer Time: 601.556 ms
Op Time: 16.0664 ms
Conv-GPU==
Layer Time: 489.933 ms
Op Time: 66.3074 ms

## use constant weight
Conv-GPU==
Layer Time: 629.162 ms
Op Time: 14.6564 ms
Conv-GPU==
Layer Time: 492.281 ms
Op Time: 61.0805 ms

## use shared memory and shared weight
Conv-GPU==
Layer Time: 616.779 ms
Op Time: 17.007 ms
Conv-GPU==
Layer Time: 504.124 ms
Op Time: 84.5762 ms

## use shared memory and constant weight
Conv-GPU==
Layer Time: 635.499 ms
Op Time: 13.9476 ms
Conv-GPU==
Layer Time: 507.771 ms
Op Time: 73.9741 ms

## use shared memory and constant weight and channel reduction with atomicAdd
Conv-GPU==
Layer Time: 664.239 ms
Op Time: 15.6144 ms
Conv-GPU==
Layer Time: 529.39 ms
Op Time: 94.4537 ms


# TILE_WIDTH = 8

## use constant weight
Conv-GPU==
Layer Time: 664.389 ms
Op Time: 29.4501 ms
Conv-GPU==
Layer Time: 507.171 ms
Op Time: 63.8308 ms

## use shared memory and shared weight
Conv-GPU==
Layer Time: 645.94 ms
Op Time: 25.3575 ms
Conv-GPU==
Layer Time: 497.66 ms
Op Time: 63.3237 ms

## use shared memory and constant weight
Conv-GPU==
Layer Time: 620.424 ms
Op Time: 22.3963 ms
Conv-GPU==
Layer Time: 488.602 ms
Op Time: 63.2799 ms

## use shared memory and constant weight and channel reduction with atomicAdd
Conv-GPU==
Layer Time: 635.244 ms
Op Time: 28.218 ms
Conv-GPU==
Layer Time: 489.56 ms
Op Time: 69.0906 ms



# Combine TILE_WIDTH = 8 and TILE_WIDTH = 16

## use shared memory and constant weight
Conv-GPU==
Layer Time: 636.381 ms
Op Time: 15.1784 ms
Conv-GPU==
Layer Time: 496.012 ms
Op Time: 58.6811 ms

## use shared memory and constant weight and channel reduction with atomicAdd
Conv-GPU==
Layer Time: 607.175 ms
Op Time: 15.6381 ms
Conv-GPU==
Layer Time: 497.037 ms
Op Time: 72.1921 ms
