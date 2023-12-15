from transformers import PretrainedConfig

class DearthConfig(PretrainedConfig):
    model_type = "dearth"
    def __init__(
        self, 
        max_token_len: int = 8192,
        vocab_size: int = None, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = None,
        n_head: int = None,
        n_kv_head: int = None,
        dim: int = None,
        dim_qk_head = None,
        hidden_dim: int = None,
        multiple_of: int = None,
        dropout_rate: float = 0.0,
        layer_init_factor: float = None,
        residual_factor: float = None, # should > 1.0
        sliding_window_size: int = 4096,
        front_window_size: int = 256,
        use_rotary: bool = True,
        rope_theta: float = 10000.0,
        use_alibi: bool = False,

        mimic_attn_layer: int = None, # 1-based, starting from the bottom; The first layer should be 1, not 0
        mimic_n_head: int = None,
        mimic_n_kv_head: int = None,
        mimic_attn_dropout: float = None,
        mimic_dim_qk_head: int = None,
        mimic_use_rotary: bool = True,
        mimic_use_alibi: bool = False,

        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.max_token_len = max_token_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.dim = dim
        self.dim_qk_head = dim_qk_head
        self.hidden_dim = hidden_dim
        if hidden_dim is None:
            self.hidden_dim = dim * 4
            print(f"hidden_dim is not specified. Set to {self.hidden_dim}")
        self.multiple_of = multiple_of
        self.dropout_rate = dropout_rate
        self.layer_init_factor = layer_init_factor
        self.residual_factor = residual_factor
        self.sliding_window_size = sliding_window_size
        self.front_window_size = front_window_size
        self.use_rotary = use_rotary
        self.rope_theta = rope_theta
        self.use_alibi = use_alibi

        self.mimic_attn_layer = mimic_attn_layer
        self.mimic_n_head = mimic_n_head
        self.mimic_n_kv_head = mimic_n_kv_head
        self.mimic_attn_dropout = mimic_attn_dropout
        self.mimic_dim_qk_head = mimic_dim_qk_head
        self.mimic_use_rotary = mimic_use_rotary
        self.mimic_use_alibi = mimic_use_alibi

        if "attn_window_size" in kwargs:
            print("Warning: attn_window_size is deprecated. Please use sliding_window_size instead !!!!!!!!!!!")
            self.sliding_window_size = kwargs["attn_window_size"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def __str__(self) -> str:
        return f"""
        max_token_len = {self.max_token_len}
        vocab_size = {self.vocab_size}
        n_layer = {self.n_layer}
        n_head = {self.n_head}
        n_kv_head = {self.n_kv_head}
        dim = {self.dim}
        dim_qk_head = {self.dim_qk_head}
        hidden_dim = {self.hidden_dim}
        multiple_of = {self.multiple_of}
        dropout_rate = {self.dropout_rate}
        layer_init_factor = {self.layer_init_factor}
        residual_factor = {self.residual_factor}
        sliding_window_size = {self.sliding_window_size}
        front_window_size = {self.front_window_size}
        use_rotary = {self.use_rotary}
        use_alibi = {self.use_alibi}

        mimic_attn_layer = {self.mimic_attn_layer}
        mimic_n_head = {self.mimic_n_head}
        mimic_n_kv_head = {self.mimic_n_kv_head}
        mimic_attn_dropout = {self.mimic_attn_dropout}
        mimic_dim_qk_head = {self.mimic_dim_qk_head}
        mimic_use_rotary = {self.mimic_use_rotary}
        mimic_use_alibi = {self.mimic_use_alibi}
        """