import torch

from dearth_model import DearthForCausalLM
from dearth_config import DearthConfig


# config = DearthConfig(
#     max_token_len=1024,
#     vocab_size=52000,
#     n_layer=24,
#     n_head=4,
#     n_kv_head=2,
#     dim=128,
#     #dim_qk_head=16,
#     #hidden_dim=256,
#     #multiple_of=16,
#     dropout_rate=0.05,
#     layer_init_factor=1.0,
#     residual_factor=2.0,
#     attn_window_size=1024,
#     use_rotary=True,
#     rope_theta=10000.0,
#     use_alibi=False,
#     front_window_size=0,

#     mimic_attn_layer = 2, # 1-based, starting from the bottom; The first layer should be 1, not 0
#     mimic_n_head = 16,
#     mimic_n_kv_head = 16,
#     mimic_attn_dropout = 0.0,
#     mimic_dim_qk_head = 32,
#     mimic_use_rotary = True,
#     mimic_use_alibi = False,
# )

config = DearthConfig(
    max_token_len=10000,
    vocab_size=32000,
    n_layer=96,
    n_head=16,
    n_kv_head=4,
    dim=1024,
    #dim_qk_head=16,
    #hidden_dim=256,
    #multiple_of=16,
    dropout_rate=0.05,
    layer_init_factor=1.0,
    residual_factor=2.0,
    attn_window_size=1024,
    use_rotary=True,
    rope_theta=10000.0,
    use_alibi=False,
    front_window_size=0,

    mimic_attn_layer = 65, # 1-based, starting from the bottom; The first layer should be 1, not 0
    mimic_n_head = 32,
    mimic_n_kv_head = 8,
    mimic_attn_dropout = 0.0,
    mimic_dim_qk_head = 32,
    mimic_use_rotary = True,
    mimic_use_alibi = False,
)

torch.set_default_device("meta")

import calflops
model = DearthForCausalLM(config).to(device="meta", dtype=torch.bfloat16)
bz = 256
seqlen = 8192
input = torch.arange(0, seqlen).unsqueeze(0).to(device="meta", dtype=torch.long)
input = input.repeat(bz, 1)
flops, macs, params = calflops.calculate_flops(model, kwargs={"input_ids": input})