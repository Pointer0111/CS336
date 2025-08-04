from jaxtyping import Float, Int
from numpy import dtype
from torch import Tensor
from torch import nn
import torch
import math
from einops import einsum


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        构造线性变换模块。
        参数:
            in_features (int): 输入的最后一维大小
            out_features (int): 输出的最后一维大小
            device (torch.device | None): 存储参数的设备
            dtype (torch.dtype | None): 参数的数据类型
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, 0, std, -3 * std, 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入应用线性变换。
        参数:
            x (torch.Tensor): 输入张量
        返回:
            torch.Tensor: 线性变换后的张量
        """
        return einsum(self.weight, x,"d_out d_in, ... d_in -> ... d_out")


class MyEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.empty((vocab_size, d_model), device=self.device, dtype=self.dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        根据token IDs从嵌入矩阵中选择对应的嵌入向量。
        参数:
            token_ids (torch.Tensor): 形状为 (batch_size, sequence_length) 的token ID张量
        返回:
            torch.Tensor: 形状为 (batch_size, sequence_length, d_model) 的嵌入向量张量
        """
        # 使用索引操作从嵌入矩阵中选择对应的嵌入向量
        # self.weight 的形状是 (vocab_size, d_model)
        # token_ids 的形状是 (batch_size, sequence_length)
        # 结果形状将是 (batch_size, sequence_length, d_model)
        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)

        # 计算RMS，保持最后一个维度以便广播！
        y = torch.sqrt(torch.mean(x**2.0, dim=-1, keepdim=True) + self.eps)

        result = x / y 

        return result.to(device=self.device, dtype=self.dtype)

def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    norm = RMSNorm(d_model, eps, 
        device=in_features.device, dtype=in_features.dtype)

    in_features = norm(in_features)


    return in_features * weights.to(in_features.device)


def silu(x: torch.Tensor):

    return torch.sigmoid(x) * x


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    in_features = in_features - in_features.max(dim=dim, keepdim=True)[0]
    exp_in_features = torch.exp(in_features)
    return exp_in_features / exp_in_features.sum(dim=dim, keepdim=True)


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    w1x = einsum(w1_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")
    w3x = einsum(w3_weight, in_features, "d_ff d_model, ... d_model -> ... d_ff")

    output = silu(w1x) * w3x

    return einsum(w2_weight, output, "d_model d_ff, ... d_ff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 预计算sin和cos值
        # 创建位置索引 [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # 频率下标 - 正确的RoPE频率计算
        freqs = self.theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = positions[:, None] / freqs[None, :]  # (max_seq_len, d_k//2)
        sin_values = torch.sin(angles)
        cos_values = torch.cos(angles)
        self.register_buffer("sin_values", sin_values, persistent=False)
        self.register_buffer("cos_values", cos_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        应用RoPE旋转到输入张量。
        
        参数:
            x (torch.Tensor): 形状为 (... seq_len, d_k) 的输入张量
            token_positions (torch.Tensor): 形状为 (... seq_len) 的token位置张量
            
        返回:
            torch.Tensor: 旋转后的张量，形状与输入相同
        """
        # 获取输入的形状信息
        original_shape = x.shape
        d_k = self.d_k
        
        # 确保d_k是偶数
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        
        # 重塑输入为 (... seq_len, d_k//2, 2)
        # 这样每对相邻的元素可以作为一个2D向量进行旋转
        x_reshaped = x.view(*original_shape[:-1], d_k // 2, 2)
        
        # 根据token_positions获取对应的sin和cos值
        # token_positions的形状: (... seq_len)
        # 需要广播到 (... seq_len, d_k//2)
        
        # 从预计算的缓冲区中获取sin和cos值
        sin_vals = self.sin_values[token_positions]  # 形状: (... seq_len, d_k//2)
        cos_vals = self.cos_values[token_positions]  # 形状: (... seq_len, d_k//2)
        
        # 应用2D旋转矩阵:
        # [cos(θ)  -sin(θ)] [x1] = [x1*cos(θ) - x2*sin(θ)]
        # [sin(θ)   cos(θ)] [x2]   [x1*sin(θ) + x2*cos(θ)]
        
        # 分离x1和x2（每对的第一个和第二个元素）
        x1 = x_reshaped[..., 0]  # 形状: (... seq_len, d_k//2)
        x2 = x_reshaped[..., 1]  # 形状: (... seq_len, d_k//2)
        
        # 应用旋转
        rotated_x1 = x1 * cos_vals - x2 * sin_vals
        rotated_x2 = x1 * sin_vals + x2 * cos_vals 
        
        # 重新组合旋转后的值
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1)  # 形状: (... seq_len, d_k//2, 2)
        
        # 重塑回原始形状
        result = rotated_x.view(original_shape)
        
        return result

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scale = 1 / math.sqrt(d_k)

    S = scale * einsum(Q, K.transpose(-2, -1), "... q d_k, ... d_k k -> ... q k")
    if mask is not None:
        S = S.masked_fill(~mask, float("-inf"))
    return einsum(softmax(S, dim=-1), V, "... q k, ... k d_v -> ... q d_v")


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    
    Q = einsum(q_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    K = einsum(k_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    V = einsum(v_proj_weight, in_features,
         "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

    d_k = K.shape[-1]
    seq_len = K.shape[-2]
    d_v = V.shape[-1]

    h = num_heads
    d_kh = d_k // h
    d_vh = d_v // h

    mask = torch.triu(torch.ones(*K.shape[:-2], seq_len, seq_len), diagonal=1).to(device=Q.device, dtype=torch.bool)
    attns = []
    for i in range(h):
        attns.append(scaled_dot_product_attention(
        Q[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        K[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        V[..., i*d_vh: min((i+1)*d_vh, d_v)], 
        mask=~mask))
    
    features = torch.concat(attns, dim=-1)

    return einsum(o_proj_weight, features,
         "d_model d_v, ... sequence_length d_v -> ... sequence_length d_model")


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:

    Q = einsum(q_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    K = einsum(k_proj_weight, in_features,
         "d_k d_in, ... sequence_length d_in -> ... sequence_length d_k")
    V = einsum(v_proj_weight, in_features,
         "d_v d_in, ... sequence_length d_in -> ... sequence_length d_v")

    d_k = K.shape[-1]
    seq_len = K.shape[-2]
    d_v = V.shape[-1]

    h = num_heads
    d_kh = d_k // h
    d_vh = d_v // h

    # 应用RoPE
    if token_positions is not None:
        max_seq_len = max_seq_len
        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_kh,
            max_seq_len=max_seq_len,
            device=in_features.device
        )

        # 对每个头分别应用RoPE
        for i in range(h):
            # 获取当前头的Q和K
            q_head = Q[..., i*d_kh:(i+1)*d_kh]
            k_head = K[..., i*d_kh:(i+1)*d_kh]
            
            # 应用RoPE
            q_head_rope = rope(q_head, token_positions)
            k_head_rope = rope(k_head, token_positions)
            
            # 更新Q和K
            Q[..., i*d_kh:(i+1)*d_kh] = q_head_rope
            K[..., i*d_kh:(i+1)*d_kh] = k_head_rope

    mask = torch.triu(torch.ones(*K.shape[:-2], seq_len, seq_len), diagonal=1).to(device=Q.device, dtype=torch.bool)
    attns = []
    for i in range(h):
        attns.append(scaled_dot_product_attention(
        Q[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        K[..., i*d_kh: min((i+1)*d_kh, d_k)], 
        V[..., i*d_vh: min((i+1)*d_vh, d_v)], 
        mask=~mask))
    
    features = torch.concat(attns, dim=-1)

    return einsum(o_proj_weight, features,
         "d_model d_v, ... sequence_length d_v -> ... sequence_length d_model")




class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights: dict[str, Tensor]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights
    
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        batch_size, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        y = x + multihead_self_attention_with_rope(self.d_model, 
            self.num_heads, self.max_seq_len, self.theta, 
            self.weights["attn.q_proj.weight"], 
            self.weights["attn.k_proj.weight"], 
            self.weights["attn.v_proj.weight"], 
            self.weights["attn.output_proj.weight"], 
            rmsnorm(self.d_model, 1e-5, self.weights["ln1.weight"], x), token_positions)

        z = y + swiglu(self.d_model, self.d_ff, 
            self.weights["ffn.w1.weight"], 
            self.weights["ffn.w2.weight"], 
            self.weights["ffn.w3.weight"], 
            rmsnorm(self.d_model, 1e-5, self.weights["ln2.weight"], y))
    
        return z
    

def extract_layer_weights(weights: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
    """
    从完整的权重字典中提取指定层的权重。
    
    参数:
        weights (dict[str, Tensor]): 完整的权重字典
        layer_idx (int): 层索引
        
    返回:
        dict[str, Tensor]: 该层对应的权重字典
    """
    layer_weights = {}
    layer_prefix = f"layers.{layer_idx}."
    
    # 提取该层的所有权重
    for key, value in weights.items():
        if key.startswith(layer_prefix):
            # 移除层前缀，保持原有的键名结构
            new_key = key[len(layer_prefix):]
            layer_weights[new_key] = value
    
    return layer_weights

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, weights: dict[str, Tensor]):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ln_final_weight = weights["ln_final.weight"]

        # 为每一层提取对应的权重
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, extract_layer_weights(weights, i)) 
            for i in range(num_layers)
        ])
        
        # 创建token embeddings并赋值权重
        self.token_embeddings = MyEmbedding(vocab_size, d_model, device=weights["token_embeddings.weight"].device, dtype=weights["token_embeddings.weight"].dtype)
        self.token_embeddings.weight.data = weights["token_embeddings.weight"]
        
        
        # 创建语言模型头并赋值权重
        self.lm_head = MyLinear(d_model, vocab_size, device=weights["lm_head.weight"].device, dtype=weights["lm_head.weight"].dtype)
        self.lm_head.weight.data = weights["lm_head.weight"]

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = rmsnorm(self.d_model, 1e-5, self.ln_final_weight, x) 
        return self.lm_head(x)

