import os
import torch
import json
from omegaconf import OmegaConf
from src.chroma.data import Protein
from model_interface import MInterface
from tqdm import tqdm

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
    parser.add_argument('--path_in', default='/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/train', type=str)
    parser.add_argument('--save_vqid_path', default='./test.jsonl', type=str)
    parser.add_argument('--level', default=20, type=int)
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
    
    file_names = os.listdir(args.path_in)

    # # ================ reconstruct ============
    for file_list in tqdm(batch_list(file_names, 32)):
        list_of_protein = []
        list_of_title = []
        for idx, file_name in enumerate(file_list):
            title = file_name.split('.')[0]
            input_pdb = os.path.join(args.path_in, file_name)
            
            try:
                protein = Protein(input_pdb, device='cuda') 
                X,C,S = protein.to_XCS()
            except:
                continue
            if X.shape[1]<5:
                continue
            if (C!=-1).sum() < 5:
                continue
            list_of_protein.append(protein)
            list_of_title.append(title)


        try:
            torch.manual_seed(0)
            with torch.no_grad():
                batch = model.batch_proteins(list_of_protein)       
                protein_all, list_vq_code = model.sample(batch=batch, cal_rmsd=False)
        except:
            print(file_list)
            continue

        with open(args.save_vqid_path, 'a+') as file:
            batch_id = batch['batch_id']
            chain_encoding = batch['chain_encoding']
            S = batch['S']
            for idx, title in enumerate(list_of_title):
                protein = protein_all[idx]
                value = {'seq': (S[batch_id==idx]).cpu().numpy().tolist(),
                    'vqid': (list_vq_code[idx]).cpu().numpy().tolist(),
                    # 'rmsd': rmsd[idx].cpu().numpy().item(),
                    'chain': chain_encoding[batch_id==idx].cpu().numpy().tolist()}

                # 创建一个包含键值对的字典
                line_dict = {title: value}
                file.write(json.dumps(line_dict) + '\n')
            

