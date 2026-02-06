import torch
from torch import nn
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    TimeEncoderLayer,
    TimeDecoderLayer,
)
from layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
    GroupedQueryAttentionLayer,
)
from layers.Norms import RMSNorm
from einops import rearrange

class Time_Language_Adapter:
    def __init__(self, placeholder) -> None:
        # 151647
        self.placeholder = placeholder
        # # 151646
        # self.emb_token = emb_token

    def _parse_prompt_mask(self, x_prompt, mask, repeat_num=1):
        b, c, l = x_prompt.shape
        pd_idx = torch.where(x_prompt == self.placeholder)[-1][0]
        x_prompt = torch.cat(
            [
                x_prompt[:, :, :pd_idx],
                torch.tensor(self.placeholder, dtype=x_prompt.dtype, device=x_prompt.device).repeat(b, c, repeat_num),
                x_prompt[:, :, pd_idx:],
            ],
            dim=-1,
        )
        mask = torch.cat(
            [
                mask[:, :, :pd_idx],
                torch.ones(b, c, repeat_num, dtype=mask.dtype, device=mask.device),
                mask[:, :, pd_idx:],
            ],
            dim=-1,
        )
        return x_prompt, mask


class Q_Former(nn.Module,Time_Language_Adapter):
    def __init__(self, configs, placeholder: int):
        nn.Module.__init__(self)
        Time_Language_Adapter.__init__(self, placeholder)
        self.token_num = configs.q_num
        self.trainable_query = nn.Parameter(torch.randn(configs.q_num, configs.d_model))
        self.causal_mask = False
        self.decoder = Decoder(
            [
                TimeDecoderLayer(
                    GroupedQueryAttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        num_kv_heads=configs.num_kv_heads,
                    ),
                    GroupedQueryAttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                        num_kv_heads=configs.num_kv_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.q_layers)
            ],
            norm_layer=RMSNorm(configs.d_model),
            projection=nn.Linear(
                configs.d_model, configs.backbone_config.hidden_size, bias=True
            ),
        )

    def forward(self, enc_out, x_prompt, mask, bsz=0):
        batch_size = enc_out.size(0)
        q = self.trainable_query.expand(batch_size, -1, -1)
        qf_out = self.decoder(q, enc_out)
        if self.token_num != qf_out.shape[1]:
            x_prompt, mask = self._parse_prompt_mask(x_prompt, mask, qf_out.shape[1]-self.token_num)
        return qf_out, x_prompt, mask


class Linear_adapter(nn.Module, Time_Language_Adapter):
    def __init__(self, configs, placeholder: int) -> None:
        nn.Module.__init__(self)
        Time_Language_Adapter.__init__(self, placeholder)
        self.token_num = max(configs.q_num, configs.seq_len // configs.patch_size)
        self.mlp = nn.Linear(configs.d_model, configs.backbone_config.hidden_size)

    def forward(self, enc_out, x_prompt, mask, bsz=0):
        la_out = self.mlp(enc_out)
        if self.token_num < la_out.shape[1]:
            x_prompt, mask = self._parse_prompt_mask(x_prompt, mask, la_out.shape[1]-self.token_num)
        return la_out, x_prompt, mask
