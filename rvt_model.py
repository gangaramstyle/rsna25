import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")

with app.setup:
    #import marimo as mo
    from math import sqrt, pi, log, prod

    import torch
    from torch import nn, einsum
    import torch.nn.functional as F
    from torch.amp import autocast

    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange

    @autocast('cuda', enabled = False)
    def rotate_every_two(x):
        x = rearrange(x, '... (d j) -> ... d j', j = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d j -> ... (d j)')

    class GEGLU(nn.Module):
        def forward(self, x):
            x, gates = x.chunk(2, dim = -1)
            return F.gelu(gates) * x

    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim, dropout = 0., use_glu = True):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim * 2 if use_glu else hidden_dim),
                GEGLU() if use_glu else nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )
        def forward(self, x):
            return self.net(x)

    class Attention(nn.Module):
        def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_rotary = True):
            super().__init__()
            inner_dim = dim_head *  heads
            self.use_rotary = use_rotary
            self.heads = heads
            self.scale = dim_head ** -0.5

            self.norm = nn.LayerNorm(dim)
            self.attend = nn.Softmax(dim = -1)
            self.dropout = nn.Dropout(dropout)

            self.to_q = nn.Linear(dim, inner_dim, bias = False)

            self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x, pos_emb, n_non_input_tokens):
            b, n, _, h = *x.shape, self.heads

            x = self.norm(x)

            q = self.to_q(x)

            qkv = (q, *self.to_kv(x).chunk(2, dim = -1))

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

            if self.use_rotary:
                # apply 3d rotary embeddings to queries and keys, excluding CLS tokens

                sin, cos = pos_emb
                dim_rotary = sin.shape[-1]

                sin = repeat(sin, 'b n d -> (b h) n d', h=h)
                cos = repeat(cos, 'b n d -> (b h) n d', h=h)

                # Separate out CLS (2) and Register tokens from the remainder of the input tokens
                (q_non_input_tokens, q), (k_non_input_tokens, k) = map(lambda t: (t[:, :n_non_input_tokens], t[:, n_non_input_tokens:]), (q, k))

                # handle the case where rotary dimension < head dimension

                (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
                q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
                q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))

                # Concat back the non input tokens
                q = torch.cat((q_non_input_tokens, q), dim = 1)
                k = torch.cat((k_non_input_tokens, k), dim = 1)

            dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
            return self.to_out(out)


@app.class_definition
class CoordinateRotaryEmbedding3D(nn.Module):
    """
    Generates 3D rotary embeddings from explicit, pre-normalized (y, x, z) coordinates.
    """
    def __init__(self, dim, max_freq=50):
        super().__init__()
        # The rotary dimension must be divisible by 6
        # (3 for the axes, 2 for the pairing in rotate_every_two)
        assert dim % 6 == 0, 'Rotary dimension `dim` must be divisible by 6 for 3D RoPE.'
        self.dim = dim
        self.max_freq = max_freq

        # Each axis (y, x, z) gets one-third of the rotary dimensions.
        dim_per_axis = self.dim // 3

        # The number of unique frequencies is half of that.
        scales = torch.linspace(1., max_freq / 2, dim_per_axis // 2)
        self.register_buffer('scales', scales)

    @autocast('cuda', enabled = False)
    def forward(self, coordinates):
        """
        Args:
            coordinates (Tensor): A tensor of shape (b, n, 3) with pre-normalized,
                                  floating-point (y, x, z) coordinates.
        """
        b, n, _ = coordinates.shape
        device, dtype = coordinates.device, self.scales.dtype

        # TODO: assert max an min values are within a certain range

        # Separate y, x, and z coordinates
        y_coords, x_coords, z_coords = coordinates.unbind(-1)

        # Reshape for broadcasting with scales
        y_norm = rearrange(y_coords, 'b n -> b n 1')
        x_norm = rearrange(x_coords, 'b n -> b n 1')
        z_norm = rearrange(z_coords, 'b n -> b n 1')

        # Calculate frequencies for each axis
        scales_dev = self.scales.to(device)
        y_freqs = y_norm * scales_dev * pi
        x_freqs = x_norm * scales_dev * pi
        z_freqs = z_norm * scales_dev * pi

        # Concatenate frequencies for y, x, and z.
        # Each part has shape (b, n, dim // 6). Concatenated shape: (b, n, dim // 2)
        all_freqs_sin = torch.cat((y_freqs.sin(), x_freqs.sin(), z_freqs.sin()), dim=-1)
        all_freqs_cos = torch.cat((y_freqs.cos(), x_freqs.cos(), z_freqs.cos()), dim=-1)

        # Reshape for the attention mechanism by interleaving.
        # (b, n, d/2) -> (b, n, d)
        sin, cos = map(lambda t: repeat(t, 'b n d -> b n (d j)', j=2), (all_freqs_sin, all_freqs_cos))

        # We need to cast back to the original input's dtype if it was e.g. float16
        return sin.to(coordinates.dtype), cos.to(coordinates.dtype)


@app.class_definition
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., use_rotary = True, use_glu = True):
        super().__init__()
        self.layers = nn.ModuleList([])

        # TODO: Are we ok with max_freq of 50? What does this correspond to in cm?
        self.pos_emb = CoordinateRotaryEmbedding3D(dim_head, max_freq = 50)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, use_rotary = use_rotary),
                FeedForward(dim, mlp_dim, dropout = dropout, use_glu = use_glu)
            ]))

    def forward(self, x, coords, n_non_input_tokens):
        # DELTA: this is different than lucid-rains implementation
        pos_emb = self.pos_emb(coords)

        for attn, ff in self.layers:
            x = attn(x, pos_emb, n_non_input_tokens) + x
            x = ff(x) + x
        return x


@app.class_definition
class RvT(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, register_count=8, dropout = 0., emb_dropout = 0., use_rotary = True, use_glu = True):
        super().__init__()

        assert dim % heads == 0, 'Model dimension `dim` must be divisible by the number of `heads`.'
        dim_head = dim // heads

        # Rotary embedding requires dim_head to be divisible by 6
        if use_rotary:
            assert dim_head % 6 == 0, 'Head dimension `dim_head` must be divisible by 6 for 3D RoPE.'

        patch_dim = prod(patch_size)

        self.patch_size = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b n pd pc pr -> b n (pd pc pr)'),
            nn.Linear(patch_dim, dim),
        )

        self.register_count=register_count

        self.cls_scan_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_token = nn.Parameter(torch.randn(1, 1, dim))
        self.register_tokens = nn.Parameter(torch.randn(1, register_count, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, use_rotary, use_glu)

    def forward(self, patches, coords):

        x = self.to_patch_embedding(patches)

        b, n = x.shape[:2]

        cls_scan_tokens = repeat(self.cls_scan_token, '() n d -> b n d', b = b)
        cls_pos_tokens = repeat(self.cls_pos_token, '() n d -> b n d', b = b)
        register_tokens = repeat(self.register_tokens, '() n d -> b n d', b = b)

        x = torch.cat((cls_scan_tokens, cls_pos_tokens, register_tokens, x), dim=1)

        x = self.transformer(x, coords, n_non_input_tokens=2+self.register_count)

        return x


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
