import os
import torch
import json
from omegaconf import OmegaConf
from src.chroma.data import Protein
from model_interface import MInterface
from tqdm import tqdm
from src.tools.utils import cuda

def safe_index(s):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return [alphabet.index(c) if c in alphabet else 0 for c in s]

def batch_list(input_list, batch_size):
    """将输入列表分成指定大小的批次"""
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]
    
def load_model(args):
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    model = MInterface(**config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda'))
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda')
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/ec_test.jsonl', type=str)
    parser.add_argument('--save_vqid_path', default='/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/EC_test_FT5_VQ16.jsonl', type=str)
    parser.add_argument('--level', default=16, type=int)
    parser.add_argument('--multichian', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    
    configs = {16: ['./foldtoken5/model_zoom/FSQ16/config.yaml', 
                    './foldtoken5/model_zoom/FSQ16/last.pth'],
               20: ['./foldtoken5/model_zoom/FSQ20/config.yaml', 
                    './foldtoken5/model_zoom/FSQ20/last.pth'],
               256: ['./foldtoken5/model_zoom/FSQ256/config.yaml', 
                    './foldtoken5/model_zoom/FSQ256/last.pth'],
               1024: ['./foldtoken5/model_zoom/FSQ1024/config.yaml', 
                    './foldtoken5/model_zoom/FSQ1024/last.pth'],
               4096: ['./foldtoken5/model_zoom/FSQ4096/config.yaml', 
                    './foldtoken5/model_zoom/FSQ4096/last.pth'],}
    args.config = configs[args.level][0]
    args.checkpoint = configs[args.level][1]
    
    model = load_model(args)

    with open(args.path_in, 'r') as f:
        lines = f.readlines()
    
    
    data_all = []
    for line in tqdm(lines):
        data = json.loads(line)
        X = torch.stack([torch.tensor(data['coords']['N']), 
                        torch.tensor(data['coords']['CA']), 
                        torch.tensor(data['coords']['C']), 
                        torch.tensor(data['coords']['O'])], dim=1)
        S = torch.tensor(safe_index(data['seq']))
        if S.shape[0]<30:
            continue
        C= []
        for cid, (cname, length) in enumerate(data['chain_length'].items()):
            C += [cid+1]*length
        C = torch.tensor(C)
        X, C, S = model.filter_nan_data(X, C, S)
        
        
        ## ========== multi-chain ============
        if args.multichian:
            name = data['name'].split('.')[0]
            if length<30:
                continue
            data_all.append((name, X, C, S))
        else:
            ## ========== single-chain ============
            for cid in C.unique():
                name = data['name'].split('.')[0] + f'_{cid}'
                mask = C==cid
                length = mask.sum()
                if length<30:
                    continue
                data_all.append((name, X[mask], C[mask], S[mask]))
        
        # if len(data_all)>1000:
        #     break
        
    all_data = []
    # # ================ reconstruct ============
    # step = 0
    for list_of_data in tqdm(batch_list(data_all, 32)):
        torch.manual_seed(0)
        with torch.no_grad():
            # batch = model.batch_data_multi_chain(list_of_data)
            # if step==10:
            #     print()
            batch = model.batch_data(list_of_data)
            batch = cuda(batch)
            h_V, vq_code = model.model(batch=batch, vq_only=True)
        
        # step+=1

        batch_id = batch['batch_id']
        chain_encoding = batch['chain_encoding']
        S = batch['S']
        for idx, title in enumerate(batch['title']):
            if args.multichian:
                value = {'seq': (S[batch_id==idx]).cpu().numpy().tolist(),
                    'vqid': (vq_code[batch_id==idx]).cpu().numpy().tolist(),
                    'chain': (chain_encoding[batch_id==idx]).cpu().numpy().tolist()}
            else:
                value = {'seq': (S[batch_id==idx]).cpu().numpy().tolist(),
                    'vqid': (vq_code[batch_id==idx]).cpu().numpy().tolist()}

            # 创建一个包含键值对的字典
            all_data.append({title: value})

    with open(args.save_vqid_path, 'a+') as file:
        for entry in all_data:
            file.write(json.dumps(entry) + '\n')
            

        


        



    