import numpy as np

mask = np.array([[0, 0, 1, 1, 1],[0, 0, 0, 1, 1]])
y_embed = mask.cumsum(0)
x_embed = mask.cumsum(1)
print("x_embed:\n", x_embed)
print("y_embed:\n", y_embed)

num_pos_feats = 6
temperature = 20
dim_t = np.arange(num_pos_feats)
dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
print("dim_t:\n", dim_t)

pos_x = x_embed[:, :, None] / dim_t
pos_y = y_embed[:, :, None] / dim_t
print("x_embed shape: ", x_embed.shape)
print("pos shape: ", pos_x.shape)
print("pos_x:\n", pos_x)
# print("pos_y:\n", pos_y)
