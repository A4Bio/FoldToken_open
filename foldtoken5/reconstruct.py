import os
import torch
from omegaconf import OmegaConf
from src.chroma.data import Protein
from model_interface import MInterface
import json

def batch_list(input_list, batch_size):
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
    parser.add_argument('--path_in', default='./N128', type=str)
    parser.add_argument('--path_out', default='./foldtoken5/N128_pred', type=str)
    # parser.add_argument('--config', default='./foldtoken5/model_zoom/FSQ20/config.yaml', type=str)
    # parser.add_argument('--checkpoint', default='./foldtoken5/model_zoom/FSQ20/last.pth', type=str)
    parser.add_argument('--level', default=4096, type=int)
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
    args.path_out = args.path_out+f'_level{args.level}/'
    os.makedirs(args.path_out, exist_ok=True)

    # # ================ reconstruct ============
    # vqids = torch.tensor([136, 141, 200, 221, 168, 76, 142, 172, 238, 125, 174, 255, 47, 171, 231, 163, 226, 218, 223, 174, 94, 157, 109, 204, 76, 136, 152, 196, 148, 232, 104, 108, 8, 140, 13, 110, 143, 175, 206, 219, 234, 230, 254, 229, 213, 210, 193, 211, 235, 201, 194, 151, 206, 133, 131, 91, 137, 129, 67, 15, 132, 66, 7, 5, 22, 25, 43, 75, 158, 73, 201, 136, 201, 217, 216, 212, 228, 234, 202, 206, 75, 77, 15, 13, 109, 30, 108, 62, 175, 107, 42, 123, 173, 45, 58, 185, 124, 44, 53, 60, 24, 42, 53, 50, 21, 30, 59, 34, 9, 31, 51, 1, 14, 107, 18, 5, 75, 83, 1, 69, 135, 82, 0, 133, 146, 139, 167, 159, 243, 175, 39, 127, 46, 239, 75, 30, 77, 44, 76, 44, 57, 21, 14, 31, 75, 139, 158, 223, 199, 218, 147, 71, 130, 65, 82, 209, 80, 160, 146, 19, 214, 219, 230, 218, 233, 250, 255, 171, 127, 179, 122, 27, 163, 99, 242, 146, 215, 247, 241, 210, 199, 131, 71, 3, 7, 35, 167, 95, 34, 119, 111, 27, 54, 191, 31, 42, 122, 126, 45, 53, 126, 109, 174, 121, 254, 181, 59, 175, 167, 251, 185, 118, 179, 249, 116, 114, 246, 249, 253, 244, 246, 250, 238, 230, 242, 187, 231, 242, 181, 107, 163, 177, 118, 99, 162, 180, 51, 186, 59, 126, 121, 57, 109, 120, 40, 42, 36, 50, 21, 47, 119, 50, 7, 107, 118, 35, 91, 183, 98, 243, 82, 71, 66, 50, 27, 6, 70, 14, 5, 34, 43, 13, 16, 38, 46, 8, 32, 57, 28, 20, 52, 60, 88, 100], device='cuda')
    # node_idx = torch.arange(vqids.shape[0], device='cuda')
    # batch_id = torch.zeros_like(node_idx, device='cuda')
    # chain_encoding = torch.ones_like(node_idx, device='cuda')
    # X_pred = model.decode(vqids, node_idx, batch_id, chain_encoding)
    # X = X_pred
    # C = chain_encoding
    # S = torch.ones_like(C)
    # protein = Protein.from_XCS(X[None], C[None], S[None])
    # protein.to('/huyuqi/xmyu/FoldToken4_share/test.pdb', seq=vqids.tolist())
    
    
    vqids = {}
    for file_list in batch_list(file_names, 32):
        list_of_protein = []
        list_of_title = []
        for idx, file_name in enumerate(file_list):
            title = file_name.split('.')[0]
            input_pdb = os.path.join(args.path_in, file_name)
            
            protein = Protein(input_pdb, device='cuda') 
            list_of_protein.append(protein)
            list_of_title.append(title)

    
        torch.manual_seed(0)
        with torch.no_grad():
            batch = model.batch_proteins(list_of_protein)       
            list_protein_pred, list_vq_code = model.sample(batch=batch)
        
        # list_protein_pred[0].to('/huyuqi/xmyu/FoldToken4_share/test.pdb', seq=list_vq_code[0].tolist())

        for idx, title in enumerate(list_of_title):
            if os.path.exists(input_pdb) == False:
                print(f'{input_pdb} does not exist, skip')
                continue
            output_prefix = args.path_out + title+'_pred.pdb'
            if os.path.exists(output_prefix):
                print(f'{output_prefix} exists, skip')
                continue
            
            list_protein_pred[idx].to(output_prefix, mask_indices=None, seq=list_vq_code[idx].tolist())
            
            vqids[title] = list_vq_code[idx].tolist()
    
    with open(args.path_out+'vqids.json', 'w') as f:
        json.dump(vqids, f)