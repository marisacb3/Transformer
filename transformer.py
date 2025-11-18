"""
transformer.py
----------
Implementación del Transformer original del paper "Attention is All You Need" (Vaswani et al., 2017)
Este módulo define un Transformer encoder-decoder completo desde cero, sin utilizar nn.Transformer de PyTorch.

Autor: María Isabel Cabrera Bermejo
Fecha: 13/11/2025
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionalEncoding(nn.Module):
    """
    To make use of the order of the sequence, they must inject some information about the relative  or absolute position of the tokens in the sequence.
    To this end, they add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.
    The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.
    They use sine and cosine functions of different frequencies. Each dimension of the positional encoding corresponds to a sinusoid.
    They apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
    """
    def __init__(
        self,
        d_model: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)    # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)    # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    """
    A fully connected feed-forward network, which is applied to each position separately and identically.
    Consists of two linear transformations witha a ReLU activation in between.
    The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048.
    
    Args:
        - d_model: (int) dimension del embedding del modelo 
        - d_ff: (int) dimension interna del feed-forward
        - dropout: (float) tasa de dropout
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        output1 = F.relu(self.linear1(x))
        output2 = self.linear2(self.dropout(output1))
        return output2

class MultiHeadAttention(nn.Module):
    """
    The input of scaled dot-product attention consists of queries and keys of dimension d_k, and values of dimension d_v
    The dot products of the queries with all keys, divide each by the square root of d_k, and apply a softmax to obtain the weights on the values
    They compute the attention on a set of queries simultaneously, packed together into a matrix Q.
    The keys and values are also packed together into matrices K and V.

    It is beneficial to linearly project the queries, keys and values h times with different, learned linear projections d_k, d_k and d_v dimenions.
    On each of these projected versions, they perform the attention function in parallel, yielding d_v dimensional output values.
    These are concatenated and once again projected, resulting in the final values.
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q = nn.Linear(in_features=d_model, out_features=d_model)
        self.k = nn.Linear(in_features=d_model, out_features=d_model)
        self.v = nn.Linear(in_features=d_model, out_features=d_model)
        self.out = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size = query.size(0)

        # Linear projections and divide in num_heads
        Q = self.q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Calcule attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply weights to values and concat heads
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # Apply final projection
        output = self.out(context)
        return output

class DecoderLayer(nn.Module):
    """
    Each layer has three sublayers. The output of each sublayers is -> LayerNorm(x + Sublayer(x))
        1. Masked multihead self-attention. To prevent positions from attending to subsequent positions
        2. Multihead self-attention which performs multihead attention over the output of the encoder stack
        3. Simple, positional-wise fully connected feed-forward network  
    Residual connection is applied around each of the three sublayers, followed by layer normalization
    All sublayers, as well as the embedding layers, produce outputs of dimension d_model = 512
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn1 = self.attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn1))

        attn2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(attn2))

        ff_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt

class Decoder(nn.Module):
    """
    Composed of a stack of N = 6 identical layers
    They use learned embeddings to convert the ouput tokens to vectors of dimension d_model
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    """
    Each layer has two sublayers. The output of each sublayers is -> LayerNorm(x + Sublayer(x))
        1. Multihead self-attention
        2. Simple, positional-wise fully connected feed-forward network
    Residual connection is applied around each of the two sublayers, followed by layer normalization
    All sublayers, as well as the embedding layers, produce outputs of dimension d_model = 512
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        attn_out = self.attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_out))

        ff_out = self.ffn(src)
        src = self.norm2(src + self.dropout2(ff_out))
        return src

class Encoder(nn.Module):
    """
    Composed of a stack of N = 6 identical layers
    They use learned embeddings to convert the input tokens to vectors of dimension d_model
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Has an encoder-decoder structure.
    They use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.output_layer = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)

        # From paper: "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation"
        self.output_layer.weight = self.decoder.embedding.weight
        # Optionally tie encoder embedding too, only if src_vocab_size == tgt_vocab_size
        if src_vocab_size == tgt_vocab_size:
            self.encoder.embedding.weight = self.decoder.embedding.weight

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        
        if tgt_mask is None:
            tgt_len = tgt.size(1)
            tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()

        memory = self.encoder(src=src, src_mask=src_mask) # maps an input sequence of symbol representations (x_1, ..., x_n) to a sequence of continuous representations z = (z_1, ..., z_n)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=src_mask) # given z, generates an output sequence (y_1, ..., y_m) of symbols one elemnt at a time
        return self.output_layer(output)