from __future__ import annotations
from functools import partial
import einops
import torch
import torch.nn as nn
from attr import dataclass
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinTensor,
)
from esm.tokenization import get_model_tokenizers
from esm.utils import encoding
from esm.utils.constants import esm3 as C
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.misc import rbf
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from typing import Union


@dataclass
class ESMOutput:
    sequence_logits: torch.Tensor
    structure_logits: torch.Tensor
    secondary_structure_logits: torch.Tensor
    sasa_logits: torch.Tensor
    function_logits: torch.Tensor
    residue_logits: torch.Tensor
    embeddings: torch.Tensor


class EncodeInputs(nn.Module):
    """
    Module for encoding input features in the ESM-3 model.

    Args:
        d_model (int): The dimensionality of the model's hidden states.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Sequence
        self.sequence_embed = nn.Embedding(64, d_model)
        # Mandatory information
        self.plddt_projection = nn.Linear(16, d_model)
        self.structure_per_res_plddt_projection = nn.Linear(16, d_model)

        # Structure
        self.structure_tokens_embed = nn.Embedding(4096 + 5, d_model)

        # "Structural" features
        self.ss8_embed = nn.Embedding(8 + 3, d_model)
        self.sasa_embed = nn.Embedding(16 + 3, d_model)

        # "Functional" features
        self.function_embed = nn.ModuleList(
            [nn.Embedding(260, d_model // 8, padding_idx=0) for _ in range(8)]
        )

        self.residue_embed = nn.EmbeddingBag(1478, d_model, mode="sum", padding_idx=0)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        average_plddt: torch.Tensor,
        per_res_plddt: torch.Tensor,
        ss8_tokens: torch.Tensor,
        sasa_tokens: torch.Tensor,
        function_tokens: torch.Tensor,
        residue_annotation_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_embed = self.sequence_embed(sequence_tokens)

        rbf_16_fn = partial(rbf, v_min=0.0, v_max=1.0, n_bins=16)
        # the `masked_fill(padding_mask.unsqueeze(2), 0)` for the two below is unnecessary
        # as pad tokens never even interact with the "real" tokens (due to sequence_id)
        plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt))
        structure_per_res_plddt = self.structure_per_res_plddt_projection(
            rbf_16_fn(per_res_plddt)
        )

        # Structure + "structural features" embeds
        structure_embed = self.structure_tokens_embed(structure_tokens)
        ss8_embed = self.ss8_embed(ss8_tokens)
        sasa_embed = self.sasa_embed(sasa_tokens)

        # "Functional" features embeds
        function_embed = torch.cat(
            [
                embed_fn(funcs)
                for embed_fn, funcs in zip(
                    self.function_embed, function_tokens.unbind(-1)
                )
            ],
            -1,
        )

        # Residue embeds
        B, L, N = residue_annotation_tokens.shape
        residue_embed = self.residue_embed(
            einops.rearrange(
                residue_annotation_tokens, "B L N -> (B L) N", B=B, L=L, N=N
            )
        )
        residue_embed = einops.rearrange(residue_embed, "(B L) D -> B L D", B=B, L=L)

        return (
            sequence_embed
            + plddt_embed
            + structure_per_res_plddt
            + structure_embed
            + ss8_embed
            + sasa_embed
            + function_embed
            + residue_embed
        )


class OutputHeads(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, 64)
        self.structure_head = RegressionHead(d_model, 4096)
        self.ss8_head = RegressionHead(d_model, 8 + 3)
        self.sasa_head = RegressionHead(d_model, 16 + 3)
        self.function_head = RegressionHead(d_model, 260 * 8)
        self.residue_head = RegressionHead(d_model, 1478)

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        sequence_logits = self.sequence_head(x)
        structure_logits = self.structure_head(x)
        secondary_structure_logits = self.ss8_head(x)
        sasa_logits = self.sasa_head(x)
        function_logits = self.function_head(x)
        function_logits = einops.rearrange(
            function_logits,
            "... (k v) -> ... k v",
            k=8,
        )

        residue_logits = self.residue_head(x)

        return ESMOutput(
            sequence_logits=sequence_logits,
            structure_logits=structure_logits,
            secondary_structure_logits=secondary_structure_logits,
            sasa_logits=sasa_logits,
            function_logits=function_logits,
            residue_logits=residue_logits,
            embeddings=embed,
        )


class ESM3(nn.Module, ESM3InferenceClient):
    """
    ESM3 model implementation.
    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        v_heads (int): The number of attention heads in the variational transformer layers.
        n_layers (int): The number of transformer layers.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int,
        n_layers: int,
        structure_encoder_name: str,
        structure_decoder_name: str,
        function_decoder_name: str,
    ):
        super().__init__()
        self.encoder = EncodeInputs(d_model)
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
        )
        self.output_heads = OutputHeads(d_model)

        self.structure_encoder_name = structure_encoder_name
        self.structure_decoder_name = structure_decoder_name
        self.function_decoder_name = function_decoder_name

        self.structure_encoder: Union[StructureTokenEncoder, None] = None  # type: ignore
        self.structure_decoder: Union[StructureTokenDecoder, None] = None  # type: ignore
        self.function_decoder: Union[FunctionTokenDecoder, None] = None  # type: ignore

        self.tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ESM3_OPEN_SMALL,
        device: Union[torch.device , str] = "cpu",
    ) -> ESM3:
        from esm.pretrained import load_local_model

        if model_name not in [ESM3_OPEN_SMALL]:
            raise ValueError(f"Model name {model_name} is not a valid ESM3 model name.")
        model: ESM3 = load_local_model(model_name, device=device)  # type: ignore
        return model
    """用了"""
    def get_structure_token_encoder(self) -> StructureTokenEncoder:
        if self.structure_encoder is None:
            self.structure_encoder = self.load_model(self.structure_encoder_name)  # type: ignore
        return self.structure_encoder  # type: ignore

    def load_model(self, model_name: str):
        # Lazy import from pretrained
        from esm.pretrained import load_local_model

        return load_local_model(model_name, device=next(self.parameters()).device)

    def forward(
        self,
        *,
        sequence_tokens: Union[torch.Tensor , None] = None,
        structure_tokens: Union[torch.Tensor , None] = None,
        ss8_tokens: Union[torch.Tensor , None] = None,
        sasa_tokens: Union[torch.Tensor , None] = None,
        function_tokens: Union[torch.Tensor , None] = None,
        residue_annotation_tokens: Union[torch.Tensor , None] = None,
        average_plddt: Union[torch.Tensor , None] = None,
        per_res_plddt: Union[torch.Tensor , None] = None,
        structure_coords: Union[torch.Tensor , None] = None,
        chain_id: Union[torch.Tensor , None] = None,
        sequence_id: Union[torch.Tensor , None] = None,
    ) -> ESMOutput:
        """
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_UNK_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_UNK_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)
        sequence_id = defaults(sequence_id, 0)

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        if structure_tokens is None:
            _, structure_tokens = self.get_structure_token_encoder().encode(
                structure_coords
            )
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(
                (structure_tokens == -1) | ~affine_mask, C.STRUCTURE_MASK_TOKEN
            )
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding = self.transformer(x, sequence_id, affine, affine_mask, chain_id)
        return self.output_heads(x, embedding)

    # 以下是针对 ESM3InferenceClient 接口的方法
    """用了"""
    def tokenizer(self, input: ESMProtein) -> ESMProteinTensor:
        # input = attr.evolve(input)  # Make a copy
        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None
        coordinates = None
    
        #if input.coordinates is not None:
        _, _, structure_tokens = encoding.tokenize_structure(
            input.coordinates,
            self.get_structure_token_encoder(),
            structure_tokenizer=self.tokenizers.structure,
            reference_sequence=input.sequence or "",
               add_special_tokens=True,
        )
        
        return ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structure_tokens,
            secondary_structure=secondary_structure_tokens,
            sasa=sasa_tokens,
            function=function_tokens,
            residue_annotations=residue_annotation_tokens,
            coordinates=coordinates,
        ).to(next(self.parameters()).device)

    