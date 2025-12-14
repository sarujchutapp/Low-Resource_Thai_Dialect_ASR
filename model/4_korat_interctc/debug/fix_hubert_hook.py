import os

TARGET_FILE = "/mnt/c/users/Saruj/espnet/espnet2/asr/encoder/hubert_encoder.py"

# The content uses a Forward Hook to capture intermediate layers
# regardless of FairSeq version argument compatibility.
new_content = '''# Copyright 2021 Tianzi Wang
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0

"""Encoder definition."""
import contextlib
import copy
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import yaml
from filelock import FileLock
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class FairseqHubertEncoder(AbsEncoder):
    """FairSeq Hubert encoder module with InterCTC support via Hooks."""

    @typechecked
    def __init__(
        self,
        input_size: int,
        hubert_url: str = "./",
        hubert_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str = "static",
        layerdrop: float = 0.1,
        feature_grad_mult: float = 0.0,
    ):
        super().__init__()
        self.apply_mask = apply_mask
        try:
            import fairseq
            from fairseq.models.hubert.hubert import HubertModel
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            raise e

        arg_overrides = {
            "dropout": dropout_rate,
            "activation_dropout": activation_dropout,
            "attention_dropout": attention_dropout,
            "mask_length": mask_length,
            "mask_prob": mask_prob,
            "mask_selection": mask_selection,
            "mask_other": mask_other,
            "mask_channel_length": mask_channel_length,
            "mask_channel_prob": mask_channel_prob,
            "mask_channel_selection": mask_channel_selection,
            "mask_channel_other": mask_channel_other,
            "encoder_layerdrop": layerdrop,
            "feature_grad_mult": feature_grad_mult,
            "data": hubert_dir_path,
        }

        if hubert_url == "espnet":
            self.hubert_model_path = hubert_dir_path
            s = torch.load(self.hubert_model_path, map_location="cpu")
            state = {k.replace("encoder.encoder.", ""): v for k, v in s.items() if "label_embs_concat" not in k}
            
            config_file = os.path.join("/".join(self.hubert_model_path.split("/")[:-1]), "config.yaml")
            with open(config_file, "r", encoding="utf-8") as f:
                self.pretrained_cfg = yaml.safe_load(f)

            model = FairseqHubertPretrainEncoder(
                input_size=self.pretrained_cfg["input_size"],
                hubert_dict=self.pretrained_cfg["hubert_dict"],
                **self.pretrained_cfg["encoder_conf"],
            )
            model = model.encoder
            d = self.pretrained_cfg["encoder_conf"]["output_size"]
            self.pretrained_params = copy.deepcopy(state)
        else:
            self.hubert_model_path = download_hubert(hubert_url, hubert_dir_path)
            models, self.pretrained_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_model_path], arg_overrides=arg_overrides, strict=False
            )
            model = models[0]
            d = self.pretrained_cfg.model.encoder_embed_dim
            self.pretrained_params = copy.deepcopy(model.state_dict())

        self._output_size = output_size
        if not isinstance(model, HubertModel):
            try:
                model = model.hubert_encoder.hubert_model
            except Exception:
                pass

        self.encoders = model
        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if output_size and output_size != d:
            self.output_layer = torch.nn.Sequential(torch.nn.Linear(d, output_size))
        else:
            self.output_layer = None

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.tensor([0], dtype=torch.long))
        
        # --- INTERCTC HOOK SETUP ---
        self.interctc_outputs = []
        def hook_fn(module, input, output):
            # output is usually (x, attn_weights) or just x
            # We want x, permuted to (Batch, Time, Dim)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            
            # FairSeq transformers usually output (Time, Batch, Dim), we need (Batch, Time, Dim)
            if x.size(0) != input[0].size(1): # Heuristic check for T vs B
                 x = x.transpose(0, 1)
                 
            self.interctc_outputs.append(x)

        # Attach hooks to FairSeq Transformer Layers
        # Usually found at self.encoders.encoder.layers
        if hasattr(self.encoders, "encoder") and hasattr(self.encoders.encoder, "layers"):
            for layer in self.encoders.encoder.layers:
                layer.register_forward_hook(hook_fn)
        elif hasattr(self.encoders, "wav2vec2") and hasattr(self.encoders.wav2vec2.encoder, "layers"):
             for layer in self.encoders.wav2vec2.encoder.layers:
                layer.register_forward_hook(hook_fn)
        # ---------------------------

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        # 1. MEMORY FIX: Linear Mask Creation
        B_dim = xs_pad.size(0)
        T_dim = xs_pad.size(1)
        masks = torch.arange(T_dim, device=xs_pad.device).view(1, -1).expand(B_dim, -1) >= ilens.view(-1, 1)

        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1
            
        # Clear previous hooks
        self.interctc_outputs = []

        # 2. STANDARD FORWARD PASS (Hooks will catch the layers automatically)
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                padding_mask=masks,
                mask=self.apply_mask and self.training,
                features_only=True,
                output_layer=None,
            )

        xs_pad = enc_outputs["x"]
        masks = enc_outputs["padding_mask"]
        del enc_outputs

        olens = (~masks).sum(dim=1)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)
        
        # 3. RETURN CAPTURED LAYERS
        # Copy list to avoid reference issues, then clear buffer
        layer_results = list(self.interctc_outputs)
        
        return xs_pad, olens, layer_results

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")

# --- Stub classes for dependency satisfaction ---
class TorchAudioHuBERTPretrainEncoder(AbsEncoder):
    def __init__(self, **kwargs): super().__init__(); pass
    def output_size(self): return 256
    def forward(self, **kwargs): return None, None, None

class FairseqHubertPretrainEncoder(AbsEncoder):
    def __init__(self, **kwargs): super().__init__(); pass
    def forward(self, **kwargs): pass
    def output_size(self): return 256

def download_hubert(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)
    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            logging.info(f"Hubert model downloaded {model_path}")
    return model_path
'''

def fix_hubert_hooks():
    if not os.path.exists(os.path.dirname(TARGET_FILE)):
        print("❌ Error: Directory does not exist.")
        return

    print(f"🔧 Overwriting {TARGET_FILE} with Hook-based InterCTC fix...")
    with open(TARGET_FILE, "w") as f:
        f.write(new_content)
    print("✅ Successfully restored hubert_encoder.py!")

if __name__ == "__main__":
    fix_hubert_hooks()