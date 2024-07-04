import marimo

__generated_with = "0.6.13"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        \"\"\"
        ---
        title: Transformer Encoder and Decoder Models
        summary: >
          These are PyTorch implementations of Transformer based encoder and decoder models,
          as well as other related modules.
        ---

        # Transformer Encoder and Decoder Models

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/basic/autoregressive_experiment.ipynb)
        \"\"\"
        """
    )
    return


@app.cell
def __():
    import math
    import torch
    import torch.nn as nn
    import import_ipynb
    from labml_nn.utils import clone_module_list
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Transformer")))
    from mha import MultiHeadAttention
    from feed_forward import FeedForward
    return (
        FeedForward,
        MultiHeadAttention,
        clone_module_list,
        import_ipynb,
        math,
        nn,
        os,
        sys,
        torch,
    )


@app.cell
def __(get_positional_encoding, math, nn, torch):
    # 位置编码： linear_embedding * \sqrt{d_model} + positional_encodings
    class EmbeddingsWithPositionalEncoding(nn.Module):
        """
        <a id="EmbeddingsWithPositionalEncoding"></a>

        ## Embed tokens and add [fixed positional encoding](positional_encoding.html)
        """

        def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
            super().__init__()
            self.linear = nn.Embedding(n_vocab, d_model)
            self.d_model = d_model
            self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

        def forward(self, x: torch.Tensor):
            pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
            return self.linear(x) * math.sqrt(self.d_model) + pe

    return EmbeddingsWithPositionalEncoding,


@app.cell
def __(math, nn, torch):
    # 可学习的位置编码，把位置编码变成(max_len, 1, d_model)的参数即可)
    class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
        """
        <a id="EmbeddingsWithLearnedPositionalEncoding"></a>

        ## Embed tokens and add parameterized positional encodings
        """

        def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
            super().__init__()
            self.linear = nn.Embedding(n_vocab, d_model)
            self.d_model = d_model
            self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

        def forward(self, x: torch.Tensor):
            pe = self.positional_encodings[:x.shape[0]]
            return self.linear(x) * math.sqrt(self.d_model) + pe
    return EmbeddingsWithLearnedPositionalEncoding,


@app.cell
def __(FeedForward, MultiHeadAttention, nn, torch):
    # 注意力_层归一 -> 自注意力 -> dropout、残差链接-> 前馈——层归一 -> 前馈层 -> dropout、残差链接
    #             -> 如果不是自注意力 -> dropout、残差链接
    class TransformerLayer(nn.Module):
        """
        <a id="TransformerLayer"></a>

        ## Transformer Layer

        This can act as an encoder layer or a decoder layer. We use pre-norm.
        """

        def __init__(self, *,
                     d_model: int,
                     self_attn: MultiHeadAttention,
                     src_attn: MultiHeadAttention = None,
                     feed_forward: FeedForward,
                     dropout_prob: float):
            """
            * `d_model` is the token embedding size
            * `self_attn` is the self attention module
            * `src_attn` is the source attention module (when this is used in a decoder)
            * `feed_forward` is the feed forward module
            * `dropout_prob` is the probability of dropping out after self attention and FFN
            """
            super().__init__()
            self.size = d_model
            self.self_attn = self_attn
            self.src_attn = src_attn
            self.feed_forward = feed_forward
            self.dropout = nn.Dropout(dropout_prob)
            self.norm_self_attn = nn.LayerNorm([d_model])
            if self.src_attn is not None:
                self.norm_src_attn = nn.LayerNorm([d_model])
            self.norm_ff = nn.LayerNorm([d_model])
            # Whether to save input to the feed forward layer
            self.is_save_ff_input = False

        def forward(self, *,
                    x: torch.Tensor,
                    mask: torch.Tensor,
                    src: torch.Tensor = None,
                    src_mask: torch.Tensor = None):
            # Normalize the vectors before doing self attention
            z = self.norm_self_attn(x)
            # Run through self attention, i.e. keys and values are from self
            self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
            # Add the self attention results
            x = x + self.dropout(self_attn)

            # If a source is provided, get results from attention to source.
            # This is when you have a decoder layer that pays attention to 
            # encoder outputs
            # 如果不是自注意力，存在源数据src，则新增attn_src，其中query为x，key、value为src
            if src is not None:
                # Normalize vectors
                z = self.norm_src_attn(x)
                # Attention to source. i.e. keys and values are from source
                attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
                # Add the source attention results
                x = x + self.dropout(attn_src)

            # Normalize for feed-forward
            z = self.norm_ff(x)
            # Save the input to the feed forward layer if specified
            # 为什么要把输入保存在前馈层
            if self.is_save_ff_input:
                self.ff_input = z.clone()
            # Pass through the feed-forward network
            ff = self.feed_forward(z)
            # Add the feed-forward results back
            x = x + self.dropout(ff)

            return x
    return TransformerLayer,


@app.cell
def __(TransformerLayer, clone_module_list, nn, torch):
    # 多层拼接 -> encoder_norm
    class Encoder(nn.Module):
        """
        <a id="Encoder"></a>

        ## Transformer Encoder
        """

        def __init__(self, layer: TransformerLayer, n_layers: int):
            super().__init__()
            # Make copies of the transformer layer
            self.layers = clone_module_list(layer, n_layers)
            # Final normalization layer
            self.norm = nn.LayerNorm([layer.size])

        def forward(self, x: torch.Tensor, mask: torch.Tensor):
            # Run through each transformer layer
            for layer in self.layers:
                x = layer(x=x, mask=mask)
            # Finally, normalize the vectors
            return self.norm(x)
    return Encoder,


@app.cell
def __(TransformerLayer, clone_module_list, nn, torch):
    # 多层拼接 -> decoder_norm
    class Decoder(nn.Module):
        """
        <a id="Decoder"></a>

        ## Transformer Decoder
        """

        def __init__(self, layer: TransformerLayer, n_layers: int):
            super().__init__()
            # Make copies of the transformer layer
            self.layers = clone_module_list(layer, n_layers)
            # Final normalization layer
            self.norm = nn.LayerNorm([layer.size])

        def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
            # Run through each transformer layer
            for layer in self.layers:
                x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
            # Finally, normalize the vectors
            return self.norm(x)
    return Decoder,


@app.cell
def __(nn):
    class Generator(nn.Module):
        """
        <a id="Generator"></a>

        ## Generator

        This predicts the tokens and gives the lof softmax of those.
        You don't need this if you are using `nn.CrossEntropyLoss`.
        """

        def __init__(self, n_vocab: int, d_model: int):
            super().__init__()
            self.projection = nn.Linear(d_model, n_vocab)

        def forward(self, x):
            return self.projection(x)
    return Generator,


@app.cell
def __(Decoder, Encoder, nn, torch):
    class EncoderDecoder(nn.Module):
        """
        <a id="EncoderDecoder"></a>

        ## Combined Encoder-Decoder
        """

        def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: nn.Module):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_embed
            self.tgt_embed = tgt_embed
            self.generator = generator

            # This was important from their code.
            # Initialize parameters with Glorot / fan_avg.
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
            # Run the source through encoder
            enc = self.encode(src, src_mask)
            # Run encodings and targets through decoder
            return self.decode(enc, src_mask, tgt, tgt_mask)

        def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
            return self.encoder(self.src_embed(src), src_mask)

        def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
            return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    return EncoderDecoder,


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()

