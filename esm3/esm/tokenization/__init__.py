from dataclasses import dataclass
from typing import Protocol

from esm.utils.constants.esm3 import VQVAE_SPECIAL_TOKENS
from esm.utils.constants.models import ESM3_OPEN_SMALL

from .structure_tokenizer import StructureTokenizer
from .tokenizer_base import EsmTokenizerBase




@dataclass
class TokenizerCollection:
    structure: StructureTokenizer
   
  


def get_model_tokenizers(model: str = ESM3_OPEN_SMALL) -> TokenizerCollection:
    if model == ESM3_OPEN_SMALL:
        return TokenizerCollection(
            structure=StructureTokenizer(vq_vae_special_tokens=VQVAE_SPECIAL_TOKENS),
        )
    else:
        raise ValueError(f"Unknown model: {model}")


# def get_invalid_tokenizer_ids(tokenizer: EsmTokenizerBase) -> list[int]:
#     if isinstance(tokenizer, EsmSequenceTokenizer):
#         return [
#             tokenizer.mask_token_id,  # type: ignore
#             tokenizer.pad_token_id,  # type: ignore
#             tokenizer.cls_token_id,  # type: ignore
#             tokenizer.eos_token_id,  # type: ignore
#         ]
#     else:
#         return [
#             tokenizer.mask_token_id,
#             tokenizer.pad_token_id,
#             tokenizer.bos_token_id,
#             tokenizer.eos_token_id,
#         ]
