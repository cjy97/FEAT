import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim = 512,
        dim_key = 128,
        dim_value = 128
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.to_qk = nn.Conv2d(dim, dim_key, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value, 1, bias = False)

    def forward(self, query_repr, supports_repr):
        """
        dimensions names:
        
        b - batch
        k - num classes
        n - num images in a support class
        c - channels
        h, i - height
        w, j - width
        """

        # b, k, *_ = img_supports.shape

        # query_repr = model(img_query)
        # *_, h, w = query_repr.shape

        # img_supports = rearrange(img_supports, 'b k n c h w -> (b k n) c h w', b = b)
        # supports_repr = model(img_supports)

        b, k, *_ = supports_repr.shape
        *_, h, w = query_repr.shape

        # print("supports_repr: ", supports_repr.size())
        supports_repr = rearrange(supports_repr, 'b k n c h w -> (b k n) c h w', b = b)
        # print("supports_repr: ", supports_repr.size())

        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)
        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)
        supports_k, supports_v = map(lambda t: rearrange(t, '(b k n) c h w -> b k n c h w', b = b, k = k), (supports_k, supports_v))

        # print("query_q: ", query_q.size())
        # print("supports_k: ", supports_k.size())

        sim = einsum('b c h w, b k n c i j -> b k h w n i j', query_q, supports_k) * self.scale
        sim = rearrange(sim, 'b k h w n i j -> b k h w (n i j)')

        attn = sim.softmax(dim = -1)
        attn = rearrange(attn, 'b k h w (n i j) -> b k h w n i j', i = h, j = w)

        out = einsum('b k h w n i j, b k n c i j -> b k c h w', attn, supports_v)

        return query_v, out


        # out = rearrange(out, 'b k c h w -> b k (c h w)')
        # query_v = rearrange(query_v, 'b c h w -> b () (c h w)')

        # euclidean_dist = ((query_v - out) ** 2).sum(dim = -1) / (h * w)
        # return -euclidean_dist
