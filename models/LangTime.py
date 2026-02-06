import numpy as np
from layers.Embed import DataEmbedding_inverted, PatchEmbedding
from layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
    GroupedQueryAttentionLayer,
)
from layers.Transformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    TimeEncoderLayer,
    TimeDecoderLayer,
)
import torch.nn.functional as F
import torch
from torch import nn
from einops import rearrange
from copy import deepcopy
import math
from layers.Adapters import Linear_adapter
from layers.Norms import RMSNorm
import time
from configs.log_config import get_logger

logger = get_logger()


class Dyanmic_MLP(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.hidden_size = configs.backbone_config.hidden_size
        self.intermediate_size = configs.backbone_config.intermediate_size
        self.mlp_dict = nn.ModuleDict(
            {
                f"mlp_{length}": TimeMLP(
                    self.hidden_size, self.intermediate_size, length
                )
                for length in configs.pretrain_seq_lens
            }
        )

    def forward(self, x, length):
        k = f"mlp_{length}"
        if k not in self.mlp_dict:
            self.mlp_dict[k] = (
                TimeMLP(self.hidden_size, self.intermediate_size, length)
                .bfloat16()
                .to(x.device)
            )
        return self.mlp_dict[k](x)


class TimeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, out_size=None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.out_size = out_size if out_size else hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Patch_enc(nn.Module):
    def __init__(self, configs):
        super(Patch_enc, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.d_model = configs.d_model
        self.causal_mask = True

        self.enc_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_size,
            stride=self.patch_size,
            dropout=0.0,
        )

        self.encoder = Encoder(
            [
                TimeEncoderLayer(
                    GroupedQueryAttentionLayer(
                        FullAttention(
                            self.causal_mask,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
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
                for l in range(configs.e_layers)
            ],
            norm_layer=RMSNorm(configs.d_model),
        )

        self.rmsnorm = RMSNorm(configs.d_model)


    def forward(self, x_enc: torch.Tensor, x_mark_enc):
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, n_vars = self.enc_embedding(x_enc)
        x_enc = self.rmsnorm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        return enc_out


TS_ENC_ARCHITECTURE = {"patch": Patch_enc}
ADAPTER_ARCHITECTURE = {"linear": Linear_adapter}


class LTModel(nn.Module):

    def __init__(
        self,
        model_args,
        placeholder: int,
        emb_token: int,
        mask_token: int,
        out_token: int,
    ) -> None:
        super().__init__()
        attn_implementation = (
            "flash_attention_2" if model_args.use_flash_attn else "eager"
        )
        llm = model_args.backbone.lower()
        if llm == "qwen2":
            from transformers.models.qwen2 import Qwen2Config, Qwen2Model
            config = Qwen2Config.from_pretrained(
                model_args.backbone_path, attn_implementation=attn_implementation, output_attentions=True
            )
            model_args.backbone_config.__dict__.update(config.__dict__)
            self.transformer = Qwen2Model.from_pretrained(
                model_args.backbone_path,
                config=model_args.backbone_config,
                torch_dtype=torch.bfloat16,
                # output_attentions=True
                # torch_dtype=torch.float32,
            )
        elif llm=="gpt2":
            from transformers.models.gpt2 import GPT2Config, GPT2Model
            config = GPT2Config.from_pretrained(model_args.backbone_path,
                                                output_attentions=True)
            model_args.backbone_config.__dict__.update(config.__dict__)
            model_args.backbone_config.hidden_size = 768
            self.transformer = GPT2Model.from_pretrained(model_args.backbone_path,
                                                         config=config)
            self.transformer.resize_token_embeddings(self.transformer.wte.weight.shape[0]+16)
            self.transformer.embed_tokens = self.transformer.wte
        else:
            raise ValueError(f"Unsupported LLM backbone: {llm}")

        self.ts_encoder = TS_ENC_ARCHITECTURE[model_args.ts_enc](model_args)

        self.adapter = ADAPTER_ARCHITECTURE[model_args.adapter_type](
            model_args, placeholder
        )

        # 151649: <|ts_out|>
        self.out_token = out_token
        # 151648: <|ts_mask|>
        self.mask_token = mask_token
        # 151647: <|TS_ENC|>
        self.placeholder = placeholder
        # 151646: <|ts_emb|>
        self.emb_token = emb_token
        self.token_num = model_args.q_num
        self.args = model_args
        self.reconfiguration(mode=model_args.training_mode)

    def reconfiguration(self, mode="full"):
        if mode == "full":
            return
        elif mode.startswith("freeze"):
            if mode == "freeze" or mode.split("_")[1] == "llm":
                self._freeze_llm(mode)
            elif mode.split("_")[1] == "adapter":
                self._freeze_adapter()
            elif mode.split("_")[1] == "enc":
                self._freeze_time_enc()
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _freeze_llm(self, mode="freeze"):
        if mode == "freeze":
            for i, (name, param) in enumerate(self.transformer.named_parameters()):
                param.requires_grad = False
        else:
            for i, (name, param) in enumerate(self.transformer.named_parameters()):
                if "norm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def _freeze_time_enc(self):
        for i, (name, param) in enumerate(self.ts_encoder.named_parameters()):
            param.requires_grad = False

    def _freeze_adapter(self):
        for i, (name, param) in enumerate(self.adapter.named_parameters()):
            param.requires_grad = False

    def random_mask(self, x_ts, mask_rate=0.0):

        max_mask_rate = 0.5
        if mask_rate == 0.0:
            return x_ts
        elif mask_rate <= 1:
            mask_rate = min(mask_rate, max_mask_rate)
        batch_size, sequence_length, num_sequences = x_ts.shape
        elements_to_mask_per_sequence = (
            int(sequence_length * mask_rate)
            if mask_rate <= 1
            else min(mask_rate, int(sequence_length * max_mask_rate))
        )
        if elements_to_mask_per_sequence == 0:
            return x_ts
        indices_to_mask = torch.randperm(sequence_length)[
            :elements_to_mask_per_sequence
        ]
        x_ts[:, indices_to_mask, :] = self.transformer.embed_tokens(
            torch.tensor(
                self.mask_token, device=self.transformer.device, dtype=torch.long
            )
        ).to(x_ts.dtype)
        return x_ts

    def _get_embed(self, x_ts, x_mark_enc, x_prompt, x_mask, mask_rate=0.0):
        """
        ts: b, len, c
        prompt: b, c, len
        """
        bsz, _, c = x_ts.shape
        # [bc, n, d]
        x_ts = self.ts_encoder(x_ts, x_mark_enc)
        # [bc, q_num, d]
        x_ts, x_prompt, x_mask = self.adapter(x_ts, x_prompt, x_mask, bsz)
        # mask x
        x_ts = self.random_mask(x_ts, mask_rate)
        pd_idx = torch.where(x_prompt == self.placeholder)[-1][0]
        emb_idx = torch.where(x_prompt == self.emb_token)[-1][0]
        out_idx = torch.where(x_prompt == self.out_token)[-1][0]
        # b,c,len -> bc, len
        x_prompt = rearrange(x_prompt, "b c l -> (b c) l")
        x_mask = rearrange(x_mask, "b c l -> (b c) l")
        # bc,len,d
        x_prompt = self.transformer.embed_tokens(x_prompt)
        x_embed = torch.cat(
            [x_prompt[:, :pd_idx, :], x_ts, x_prompt[:, pd_idx + x_ts.shape[1] :, :]],
            dim=1,
        )
        # bc, len, d
        x_embed = self.transformer(inputs_embeds=x_embed, attention_mask=x_mask)
        # bc, 1, d
        y_embed = x_embed.last_hidden_state[:, out_idx - 1 : out_idx, :]
        y_embed = rearrange(y_embed, "(b c) 1 d -> b c d", b=bsz, c=c)
        # bc, 1, d
        x_embed = x_embed.last_hidden_state[:, emb_idx - 1 : emb_idx, :]
        x_embed = rearrange(x_embed, "(b c) 1 d -> b c d", b=bsz, c=c)
        return x_embed, y_embed


class LTPratrainedModel(LTModel):
    def __init__(self, model_args, placeholder, emb_token, mask_token, out_token):
        super().__init__(model_args, placeholder, emb_token, mask_token, out_token)
        self.x_out_proj = Dyanmic_MLP(model_args)
        ### full series prediction
        self.y_out_proj = TimeMLP(
            model_args.backbone_config.hidden_size,
            model_args.backbone_config.intermediate_size,
            model_args.single_pred_len,
        )

    def forward(self, x_ts, x_mark_enc, x_prompt, x_mask, mask_rate=0.0):
        length = x_ts.shape[1]
        # Normalization from Non-stationary Transformer
        means = x_ts.mean(1, keepdim=True).detach()
        x_ts = x_ts - means
        stdev = torch.sqrt(torch.var(x_ts, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_ts = x_ts / stdev

        x_embed, y_embed = self._get_embed(
            x_ts, x_mark_enc, x_prompt, x_mask, mask_rate
        )

        # b,c,d->b,c,len
        x_embed = self.x_out_proj(x_embed, length=length)
        y_embed = self.y_out_proj(y_embed)

        out = torch.cat([x_embed, y_embed], dim=-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        out = out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, length + self.args.single_pred_len, 1
        )
        out = out + means[:, 0, :].unsqueeze(1).repeat(
            1, length + self.args.single_pred_len, 1
        )
        return out

