

'''
'seq': 'AFEKLMYAPVY',
'coords': {'N': [n,3],
            'CA': [n,3],
            'C': [n,3],
            'O': [n,3]},
'chain_length': {'A': 10, 'B': 10},
'name': '1a0a.pdb',
'''


# from biotite.structure.io import pdb
# import biotite.structure as struc

# # 读取 PDB 文件
# pdb_file = '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/test/1A0E-A_11110.pdb'  # 替换为你的文件名
# file = pdb.PDBFile.read(pdb_file)
# structure = file.get_structure()[0]  # 由于PDB文件可能包含多个模型，因此只取第一个模型

# # 提取骨干原子
# backbone_atoms = structure[struc.filter_backbone(structure)]

# # 打印骨干原子的坐标
# for atom in backbone_atoms:
#     print(atom.coord)

import os
from tqdm import tqdm
from src.chroma.data import Protein
from src.tools.utils import pmap_multi
import json

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/test', type=str)
    parser.add_argument('--path_sv', default='/huyuqi/xmyu/FoldToken4_share/ec_test.jsonl', type=str)
    args = parser.parse_args()
    return args


def read_pdb(file_name, path_in):
    try:
        input_pdb = os.path.join(path_in, file_name)
        protein = Protein(input_pdb, device='cpu') 
        X,C,S = protein.to_XCS()
        mask = C>0
        X = X[mask]
        C = C[mask]
        S = S[mask]
        chain_length = {}
        for idx, c in enumerate(C.unique()):
            chain_length[idx] = (C==c).sum().item()
        
        
        data = {'seq': ''.join([AA20[one] for one in S]), 
                'coords': 
                    {'N':X[:,0].tolist(),
                    'CA':X[:,1].tolist(),
                    'C':X[:,2].tolist(),
                    'O':X[:,3].tolist()}, 
                'chain_length': chain_length, 
                'name': file_name}
        return data
    except:
        return None

if __name__ == '__main__':
    args = parse_args()
    AA20 = "ACDEFGHIKLMNPQRSTVWYU"

    file_names = os.listdir(args.path_in)
    file_names = [[one] for one in file_names]
    all_data = pmap_multi(read_pdb, file_names, path_in=args.path_in)
    all_data = [one for one in all_data if one is not None]
    print(f'Number of proteins: {len(all_data)}')
    with open(args.path_sv, 'w') as f:
        for data in all_data:
            f.write(json.dumps(data)+'\n')
    print()

