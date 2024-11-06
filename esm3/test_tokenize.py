from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein
import os
from tqdm import tqdm
from esm.utils.pmap import pmap_multi
from esm.utils import encoding
import torch
import torch.nn.functional as F
from esm.utils.structure.normalize_coordinates import (
    normalize_coordinates,
)

"""命令行输入输出"""
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', default='/huyuqi/xmyu/FoldToken4_share/N128/', type=str)
    parser.add_argument('--path_out', default='/huyuqi/xmyu/FoldToken4_share/esm3/esm/path_out/', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    file_names = os.listdir(args.path_in)
    os.makedirs(args.path_out, exist_ok=True)
    #登录huggingface下载模型权重
    login('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')
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"
    #编码器设置
    structure_encoder = model.get_structure_token_encoder()


    # # ============ 以下部分要改为实现并行 ========
    # all_proteins = []
    # for file_name in tqdm(file_names):
    #     input_pdb = os.path.join(args.path_in, file_name)
    #     if os.path.exists(input_pdb) == False:
    #         print(f'{input_pdb} does not exist, skip')
    #         continue
    #     output_prefix = args.path_out + file_name+'_pred.pdb'
    #     if os.path.exists(output_prefix):
    #         print(f'{output_prefix} exists, skip')
    #         continue
    #     protein = ESMProtein.from_pdb(input_pdb)
    #     all_proteins.append(protein)
    # # ============
    
    
    ## ============并行读取pdb
    def read_pdb(file_name, path_in):
        input_pdb = os.path.join(path_in, file_name)
        protein = ESMProtein.from_pdb(input_pdb)
        return protein
    all_proteins = pmap_multi(read_pdb, [[one] for one in file_names], path_in=args.path_in)
    all_proteins = [one for one in all_proteins if one is not None]

    #原先输入默认是pdb文件，改成下面后面的注释
    #原先输入的是一个protein对象，而现在是一个protein对象的列表。所以会报错
    #result = model.tokenizer(all_proteins) # [batchsize, L, number of atom(估计是3), 3]
    maxL = 0
    coords = [ ]
    for protein in all_proteins[:10]:
        coords.append(protein.coordinates)
        if protein.coordinates.size(0) > maxL:
            maxL = protein.coordinates.size(0)
    
    coords = [F.pad(one, (0,0,0,0,0,maxL-one.shape[0]), value=torch.inf) for one in coords]
    coords = torch.stack(coords, dim=0)
    coords = normalize_coordinates(coords).cuda()
    residue_index = torch.arange(coords.shape[1], device=coords.device)
    residue_index = residue_index[None].repeat(coords.shape[0], 1)
    _, structure_tokens = structure_encoder.encode(
        coords, residue_index=residue_index
    )
    
    # for protein in all_proteins[:10]:
    #     _, _, structure_tokens2 = encoding.tokenize_structure(
    #         protein.coordinates,
    #         model.get_structure_token_encoder(),
    #         structure_tokenizer=model.tokenizers.structure,
    #         reference_sequence= "",
    #         add_special_tokens=False,
    #     )
    #     print(structure_tokens2)
    
    # results.append(result)
    # result.coordinates = None
    # result.sequence = None
    # print("pretrain is OK!")

