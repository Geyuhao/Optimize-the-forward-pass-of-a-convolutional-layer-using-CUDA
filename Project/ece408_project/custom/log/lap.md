# TILE_WIDTH = 16
## use shared memory and constant weight
Conv-GPU==
Layer Time: 635.499 ms
Op Time: 13.9476 ms
Conv-GPU==
Layer Time: 507.771 ms
Op Time: 73.9741 ms


# TILE_WIDTH = 8
## use shared memory and constant weight
Conv-GPU==
Layer Time: 620.424 ms
Op Time: 22.3963 ms
Conv-GPU==
Layer Time: 488.602 ms
Op Time: 63.2799 ms

# Combine TILE_WIDTH = 8 and TILE_WIDTH = 16

## use shared memory and constant weight
Conv-GPU==
Layer Time: 636.381 ms
Op Time: 15.1784 ms
Conv-GPU==
Layer Time: 496.012 ms
Op Time: 58.6811 ms

## use shared memory and constant weight and use overlap method (times = 1, #stream = 2) (exclisive)
Conv-GPU==
Layer Time: 624.443 ms
Op Time: 0.007161 ms
Conv-GPU==
Layer Time: 455.768 ms
Op Time: 0.007591 ms

## use shared memory and constant weight and use overlap method (times = 1, #stream = 5) (exclusive)
Conv-GPU==
Layer Time: 624.957 ms
Op Time: 0.007546 ms
Conv-GPU==
Layer Time: 463.549 ms
Op Time: 0.008064 ms

## use shared memory and constant weight and use overlap method (times = 5, #stream = 5) (exclusive)
Conv-GPU==
Layer Time: 636.319 ms
Op Time: 0.0064 ms
Conv-GPU==
Layer Time: 438.041 ms
Op Time: 0.00747 ms

## use shared memory and constant weight and use overlap method (times = 10, #stream = 5) (exclusive) !!
Conv-GPU==
Layer Time: 588.248 ms
Op Time: 0.005523 ms
Conv-GPU==
Layer Time: 429.997 ms
Op Time: 0.006553 ms

## use shared memory and constant weight and use overlap method (times = 1, #stream = 8)
Conv-GPU==
Layer Time: 637.437 ms
Op Time: 0.006921 ms
Conv-GPU==
Layer Time: 455.5 ms
Op Time: 0.00804 ms

## use shared memory and constant weight and use overlap method (times = 5, #stream = 8)
Conv-GPU==
Layer Time: 633.52 ms
Op Time: 0.006061 ms
Conv-GPU==
Layer Time: 453.296 ms
Op Time: 0.007584 ms

## use shared memory and constant weight and channel reduction with atomicAdd and use overlap method (times = 1, #stream = 8)
Conv-GPU==
Layer Time: 642.754 ms
Op Time: 0.009252 ms
Conv-GPU==
Layer Time: 478.679 ms
Op Time: 0.009232 ms

## use shared memory and constant weight and channel reduction with atomicAdd and use overlap method (times = 5, #stream = 8)
Conv-GPU==
Layer Time: 639.978 ms
Op Time: 0.006693 ms
Conv-GPU==
Layer Time: 458.578 ms
Op Time: 0.00717 ms
