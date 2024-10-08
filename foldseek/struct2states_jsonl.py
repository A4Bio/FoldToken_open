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
        return model(torch.tensor(x, dtype=torch.float32, device='cuda')).detach().cpu().numpy()

def safe_index(s):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return [alphabet.index(c) if c in alphabet else 0 for c in s]

def discretize(encoder, centroids, x):
    z = predict(encoder, x)
    return np.argmin(extract_pdb_features.distance_matrix(z, centroids), axis=1)

def batch_list(input_list, batch_size):
    """将输入列表分成指定大小的批次"""
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

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
    arg.add_argument('--file_path', default ='/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb/pdb.jsonl', type=str, help='path to jsonl')
    arg.add_argument('--save_vqid_path', default ='/huyuqi/xmyu/FoldToken4_share/pdb_foldseek.jsonl', type=str, help='path to save vqids')
    arg.add_argument('--virt', default=[270, 0, 2],type=float, nargs=3, help='virtual center')
    arg.add_argument('--invalid-state', type=str, help='for missing coords.',
        default='X')
    arg.add_argument('--exclude-feat', type=int, help='Do not calculate feature no.',
        default=None)
    args = arg.parse_args()


    encoder = torch.load(args.encoder).cuda()
    centroids = np.loadtxt(args.centroids)

    with open(args.file_path, 'r') as json_file:
        lines = json_file.readlines()
    
    data_all = []
    for line in tqdm(lines):
        data = json.loads(line)
        N = torch.tensor(data['coords']['N'])
        C = torch.tensor(data['coords']['C'])
        CA = torch.tensor(data['coords']['CA'])
        CB = approx_c_beta_position(CA, N, C)
            
        coords = torch.stack([CA, CB, N, C], dim=1)
        valid_mask = ~torch.isnan(coords.sum(dim=(1,2)))
        
        S = torch.tensor(safe_index(data['seq']))
        start_idx = 0
        for cid, (cname, length) in enumerate(data['chain_length'].items()):
            name = data['name'].split('.')[0] + f'_{cid}'
            C = torch.tensor([cid]*length)
            if length<30:
                continue
            # protein = Protein.from_XCS(X = X[start_idx:start_idx+length][None], S = S[start_idx:start_idx+length][None], C=C[None])
            # protein.sys.name = name
            data_all.append((name, coords[start_idx:start_idx+length], valid_mask[start_idx:start_idx+length], S[start_idx:start_idx+length], length))
            start_idx += length
        # if len(data_all)>1000:
        #     break
    
    all_data = []
    for list_of_data in tqdm(batch_list(data_all, 64)):
        name, coords, valid_mask, S, length = zip(*list_of_data)
        coords = torch.cat(coords, dim=0)
        valid_mask = torch.cat(valid_mask, dim=0)
        S = torch.cat(S, dim=0)
        feat, mask = create_vqvae_training_data.encoder_features_jsonl(coords.reshape(coords.shape[0],-1).numpy(), valid_mask.numpy())
        all_states = discretize(encoder, centroids, feat)
        mask = torch.from_numpy(mask[None])
        
        start_idx = 0
        for i in range(len(length)):
            value = {'seq': S[start_idx:start_idx+length[i]].tolist(),
                'vqid': all_states[start_idx:start_idx+length[i]].tolist()}
            all_data.append({name[i]: value})
            start_idx += length[i]
        # value = {'seq': S, 'vqid': [0]+valid_states.tolist()+[0]}
        
        
    
    with open(args.save_vqid_path, 'a+') as file:
        for entry in all_data:
            file.write(json.dumps(entry) + '\n')

    # data_dict = {}
    # for line in tqdm(lines):
    #     data = json.loads(line)
    #     try:
    #         N = torch.tensor(data['coords']['N'])
    #         C = torch.tensor(data['coords']['C'])
    #         CA = torch.tensor(data['coords']['CA'])
    #         CB = approx_c_beta_position(CA, N, C)
    #         coords = torch.stack([CA, CB, N, C], dim=1)
    #         valid_mask = ~torch.isnan(coords.sum(dim=(1,2)))
            
    #         chain = []
    #         cid = 1
    #         for c, l in data['chain_length'].items():
    #             chain.extend([cid for k in range(l)])
    #             cid += 1
    #         chain = torch.tensor(chain)
            
    #         feat, mask = create_vqvae_training_data.encoder_features_jsonl(coords.reshape(N.shape[0],-1).numpy(), valid_mask.numpy())
    #         # mask = C==1

    #         if args.exclude_feat is not None:
    #             fmask = np.ones(feat.shape[1], dtype=bool)
    #             fmask[args.exclude_feat - 1] = 0
    #             feat = feat[:, fmask]
    #         valid_states = discretize(encoder, centroids, feat[mask])
            
    #         mask = torch.from_numpy(mask[None])
    #         value = {'seq': [safe_indexing(s) for s in data['seq']],
    #             'vqid': valid_states.tolist(),
    #             'chain': (chain[mask[0]]).cpu().numpy().tolist()}

    #         # 创建一个包含键值对的字典
    #         data_dict.update({data['name']: value})
    #     except Exception:
    #         print(f'Error: {line}')
    #         continue

    #     # states = np.full(len(mask), -1)
    #     # states[mask] = valid_states
    # with open(args.save_vqid_path, 'a+') as file:
    #     for key, value in data_dict.items():
    #         file.write(json.dumps({key:value}) + '\n')
        