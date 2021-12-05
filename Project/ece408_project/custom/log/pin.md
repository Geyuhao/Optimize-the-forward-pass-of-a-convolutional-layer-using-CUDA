# Combine TILE_WIDTH = 8 and TILE_WIDTH = 16 
# use shared memory and constant weight and use overlap method (times = 10, #stream = 5)
Conv-GPU==
Layer Time: 358.104 ms
Op Time: 167.179 ms
Conv-GPU==
Layer Time: 258.219 ms
Op Time: 123.013 ms


# Combine TILE_WIDTH = 8 and TILE_WIDTH = 16 
# use constant weight and use overlap method (times = 10, #stream = 5) !!!
Conv-GPU==
Layer Time: 260.575 ms
Op Time: 83.5172 ms
Conv-GPU==
Layer Time: 187.767 ms
Op Time: 62.008 ms

# Combine TILE_WIDTH = 8 and TILE_WIDTH = 16 
