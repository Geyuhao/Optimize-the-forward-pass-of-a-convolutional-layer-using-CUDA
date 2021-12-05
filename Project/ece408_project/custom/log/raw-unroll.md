# TILE_WIDTH1 = 32
Conv-GPU==
Layer Time: 770.7 ms
Op Time: 151.943 ms
Conv-GPU==
Layer Time: 683.44 ms
Op Time: 193.975 ms

# TILE_WIDTH1 = 16 (exclusive)
Conv-GPU==
Layer Time: 719.469 ms
Op Time: 144.495 ms
Conv-GPU==
Layer Time: 634.149 ms
Op Time: 224.985 ms

# TILE_WIDTH1 = 8
Conv-GPU==
Layer Time: 798.17 ms
Op Time: 152.723 ms
Conv-GPU==
Layer Time: 638.794 ms
Op Time: 200.99 ms

# TILE_WIDTH1 = 16 and constant weight (exclusive)
Conv-GPU==
Layer Time: 769.102 ms
Op Time: 145.8 ms
Conv-GPU==
Layer Time: 652.157 ms
Op Time: 222.194 ms

# TILE_WIDTH1 = 16 and overlap with parellel=2 (exclusive)
Conv-GPU==
Layer Time: 957.428 ms
Op Time: 0.003276 ms
Conv-GPU==
Layer Time: 759.493 ms
Op Time: 0.002342 ms

# TILE_WIDTH1 = 16 and overlap with parellel=8 (exclusive)
Conv-GPU==
Layer Time: 923.648 ms
Op Time: 0.003922 ms
Conv-GPU==
Layer Time: 720.793 ms
Op Time: 0.003922 ms