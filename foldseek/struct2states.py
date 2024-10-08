#! /usr/bin/env python3
"""
Converts structures into 3Di sequences.

echo 'd12asa_' | ./struct2states.py encoder_.pt states_.txt --pdb_dir data/pdbs --virt 270 0 2
"""

import numpy as np
import os.path
import argparse

import torch

import create_vqvae_training_data
import extract_pdb_features
from src.chroma.data import Protein
import json
from tqdm import tqdm

# 50 letters (X/x are missing)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz'
DISTANCE_ALPHA_BETA = 1.5336
def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).detach().numpy()


def discretize(encoder, centroids, x):
    z = predict(encoder, x)
    return np.argmin(extract_pdb_features.distance_matrix(z, centroids), axis=1)


def approx_c_beta_position(c_alpha, n, c_carboxyl):
    """
    Approximate C beta position,
    from C alpha, N and C positions,
    assuming the four ligands of the C alpha
    form a regular tetrahedron.
    """
    v1 = c_carboxyl - c_alpha
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = n - c_alpha
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    b1 = v2 + 1/3 * v1
    b2 = torch.cross(v1, b1)

    u1 = b1/torch.norm(b1, dim=-1, keepdim=True)
    u2 = b2/torch.norm(b2, dim=-1, keepdim=True)

    # direction from c_alpha to c_beta
    v4 = -1/3 * v1 + np.sqrt(8)/3 * torch.norm(v1, dim=-1, keepdim=True) * (-1/2 * u1 - np.sqrt(3)/2 * u2)

    return c_alpha + DISTANCE_ALPHA_BETA * v4  # c_beta


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--encoder', default = 'foldseek/foldseek_v1/encoder.pt', type=str, help='a *.pt file')
    arg.add_argument('--centroids', default = 'foldseek/foldseek_v1/states.txt', type=str, help='np.loadtxt')
    arg.add_argument('--pdb_dir', default ='/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/test', type=str, help='path to PDBs')
    arg.add_argument('--save_vqid_path', default ='/huyuqi/xmyu/FoldToken4_share/test_foldseek.jsonl', type=str, help='path to save vqids')
    arg.add_argument('--virt', default=[270, 0, 2],type=float, nargs=3, help='virtual center')
    arg.add_argument('--invalid-state', type=str, help='for missing coords.',
        default='X')
    arg.add_argument('--exclude-feat', type=int, help='Do not calculate feature no.',
        default=None)
    args = arg.parse_args()


    encoder = torch.load(args.encoder)
    centroids = np.loadtxt(args.centroids)

    data_dict = {}
    for fn in tqdm(os.listdir(args.pdb_dir)):
        # fn = line.rstrip('\n')
        try:
            protein = Protein(args.pdb_dir + '/' + fn, device='cpu') 
            X, C, S = protein.to_XCS() # N, CA, C, O
            # CB = approx_c_beta_position(X[0, :, 1], X[0, :, 0], X[0, :, 2])
            
            feat, mask = create_vqvae_training_data.encoder_features(args.pdb_dir + '/' + fn, args.virt)
            # mask = C==1

            if args.exclude_feat is not None:
                fmask = np.ones(feat.shape[1], dtype=bool)
                fmask[args.exclude_feat - 1] = 0
                feat = feat[:, fmask]
            valid_states = discretize(encoder, centroids, feat[mask])
            title = fn.split('.')[0]
            mask = torch.from_numpy(mask[None])
            value = {'seq': (S[mask]).cpu().numpy().tolist(),
                'vqid': valid_states.tolist(),
                'chain': (C[mask]).cpu().numpy().tolist()}

            # 创建一个包含键值对的字典
            data_dict.update({title: value})
        except Exception:
            print(f'Error: {fn}')
            continue

        # states = np.full(len(mask), -1)
        # states[mask] = valid_states
    with open(args.save_vqid_path, 'a+') as file:
        file.write(json.dumps(data_dict) + '\n')
        