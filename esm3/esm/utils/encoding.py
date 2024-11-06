from typing import Sequence
import torch
import torch.nn.functional as F
from esm.models.vqvae import StructureTokenEncoder
from esm.tokenization.structure_tokenizer import (
    StructureTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain


"""用了"""
def tokenize_structure(
    coordinates: torch.Tensor,
    structure_encoder: StructureTokenEncoder,
    structure_tokenizer: StructureTokenizer,
    reference_sequence: str = "",
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(structure_encoder.parameters()).device
    chain = ProteinChain.from_atom37(
        coordinates, sequence=reference_sequence if reference_sequence else None
    )

    # Setup padding
    if reference_sequence and len(reference_sequence) != coordinates.size(0):
        raise ValueError(
            f"Reference sequence length ({len(reference_sequence)}) does not match the number of residues in the coordinates ({coordinates.size(0)})"
        )

    left_pad = 0
    right_pad = 0

    if add_special_tokens:
        left_pad += 1  # Add space for BOS token
        right_pad += 1  # Add space for EOS token

    coordinates, plddt, residue_index = chain.to_structure_encoder_inputs()
    coordinates = coordinates.to(device)  # (1, L, 37, 3)
    plddt = plddt.to(device)  # (1, L)
    residue_index = residue_index.to(device)  # (1, L)
    _, structure_tokens = structure_encoder.encode(
        coordinates, residue_index=residue_index
    )
    coordinates = torch.squeeze(coordinates, dim=0)  # (L, 37, 3)  # type: ignore
    plddt = torch.squeeze(plddt, dim=0)  # (L,)  # type: ignore
    structure_tokens = torch.squeeze(structure_tokens, dim=0)  # (L,)  # type: ignore

    # Add space for BOS and EOS tokens
    if add_special_tokens:
        coordinates = F.pad(
            coordinates,
            (0, 0, 0, 0, left_pad, right_pad),
            value=torch.inf,
        )
        plddt = F.pad(plddt, (left_pad, right_pad), value=0)
        structure_tokens = F.pad(
            structure_tokens,
            (left_pad, right_pad),
            value=structure_tokenizer.pad_token_id,
        )
        structure_tokens[0] = structure_tokenizer.bos_token_id
        structure_tokens[-1] = structure_tokenizer.eos_token_id
    return coordinates, plddt, structure_tokens


def get_default_structure_tokens(
    sequence_length: int, structure_tokenizer: StructureTokenizer
) -> torch.Tensor:
    structure_tokens = (
        torch.ones(
            (sequence_length + 2,),
            dtype=torch.int64,
        )
        * structure_tokenizer.pad_token_id
    )
    # Always include BOS and EOS tokens
    structure_tokens[0] = structure_tokenizer.bos_token_id
    structure_tokens[-1] = structure_tokenizer.eos_token_id
    return structure_tokens


