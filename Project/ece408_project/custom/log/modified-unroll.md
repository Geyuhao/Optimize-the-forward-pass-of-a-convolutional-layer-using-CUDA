# TILE_WIDTH1 = 16 num_pic = 8 (exclusive)
Conv-GPU==
Layer Time: 657.007 ms
Op Time: 68.4145 ms
Conv-GPU==
Layer Time: 494.634 ms
Op Time: 64.4457 ms

# TILE_WIDTH1 = 16 num_pic = 10 (exclusive)
Conv-GPU==
Layer Time: 650.946 ms
Op Time: 64.8979 ms
Conv-GPU==
Layer Time: 494.824 ms
Op Time: 61.598 ms

# TILE_WIDTH1 = 16 num_pic = 16 (exclusive) !!!
Conv-GPU==
Layer Time: 633.758 ms
Op Time: 66.7534 ms
Conv-GPU==
Layer Time: 473.801 ms
Op Time: 58.247 ms

# TILE_WIDTH1 = 16 num_pic = 32 (exclusive)
Conv-GPU==
Layer Time: 638.255 ms
Op Time: 75.2878 ms
Conv-GPU==
Layer Time: 473.353 ms
Op Time: 68.943 ms

# TILE_WIDTH1 = 32 num_pic = 32 (exclusive)
Conv-GPU==
Layer Time: 693.808 ms
Op Time: 74.8055 ms
Conv-GPU==
Layer Time: 494.275 ms
Op Time: 70.3866 ms

# TILE_WIDTH1 = 32 num_pic = 16 (exclusive)
Conv-GPU==
Layer Time: 658.365 ms
Op Time: 70.1381 ms
Conv-GPU==
Layer Time: 535.466 ms
Op Time: 65.1464 ms

# TILE_WIDTH1 = 16 num_pic = 8 with par = 2 (exclusive)
Conv-GPU==
Layer Time: 819.367 ms
Op Time: 0.003259 ms
Conv-GPU==
Layer Time: 625.83 ms
Op Time: 0.003097 ms

# TILE_WIDTH1 = 16 num_pic = 16 with par = 5 (exclusive) !!!
Conv-GPU==
Layer Time: 655.77 ms
Op Time: 0.003347 ms
Conv-GPU==
Layer Time: 518.574 ms
Op Time: 0.003505 ms

# TILE_WIDTH1 = 16 num_pic = 20 with par = 5 (exclusive)
Conv-GPU==
Layer Time: 710.308 ms
Op Time: 0.003296 ms
Conv-GPU==
Layer Time: 556.049 ms
Op Time: 0.003974 ms



# TILE_WIDTH2 = 64 num_pic = 16 TILE_WIDTH1 = 16
Conv-GPU==
Layer Time: 67.2357 ms
Op Time: 6.60326 ms
Conv-GPU==
Layer Time: 51.5728 ms
Op Time: 6.05375 ms


# TILE_WIDTH2 = 32 num_pic = 16 TILE_WIDTH1 = 16
Conv-GPU==
Layer Time: 65.6214 ms
Op Time: 6.34304 ms
Conv-GPU==
Layer Time: 49.7793 ms
Op Time: 5.92009 ms

# TILE_WIDTH2 = 32 num_pic = 32 TILE_WIDTH1 = 16
Conv-GPU==
Layer Time: 64.4315 ms
Op Time: 7.11353 ms
Conv-GPU==
Layer Time: 49.0856 ms
Op Time: 6.74498 ms

# TILE_WIDTH1 = 32 num_pic = 16 with par = 5 TILEWIDTH2 = 64
Conv-GPU==
Layer Time: 695.616 ms
Op Time: 0.003298 ms
Conv-GPU==
Layer Time: 524.829 ms
Op Time: 0.004088 ms

# TILE_WIDTH1 = 16 num_pic = 16 with par = 5 TILEWIDTH2 = 64
Conv-GPU==
Layer Time: 742.857 ms
Op Time: 0.004122 ms
Conv-GPU==
Layer Time: 576.707 ms
Op Time: 0.004002 ms