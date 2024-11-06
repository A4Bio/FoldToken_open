CUDA_VISIBLE_DEVICES=0 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid256.jsonl' --level 256 --multichian 0

CUDA_VISIBLE_DEVICES=1 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid16.jsonl' --level 16 --multichian 0

CUDA_VISIBLE_DEVICES=2 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid20.jsonl' --level 20 --multichian 0

CUDA_VISIBLE_DEVICES=3 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid1024.jsonl' --level 1024 --multichian 0

CUDA_VISIBLE_DEVICES=4 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid4096.jsonl' --level 4096 --multichian 0

CUDA_VISIBLE_DEVICES=5 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid256_mc.jsonl' --level 256 --multichian 1

CUDA_VISIBLE_DEVICES=6 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid16_mc.jsonl' --level 16 --multichian 1

CUDA_VISIBLE_DEVICES=7 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid20_mc.jsonl' --level 20 --multichian 1


CUDA_VISIBLE_DEVICES=0 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid1024_mc.jsonl' --level 1024 --multichian 1

CUDA_VISIBLE_DEVICES=1 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid4096_mc.jsonl' --level 4096 --multichian 1



CUDA_VISIBLE_DEVICES=0 python foldtoken5/extract_vq_ids_jsonl.py --save_vqid_path  '/huyuqi/xmyu/FoldToken4_share/foldtoken5/pdb_vqid256.jsonl' --level 20 --multichian 0



CUDA_VISIBLE_DEVICES=0 python foldtoken5/extract_vq_ids_jsonl.py --path_in  '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/ec_train.jsonl' --save_vqid_path '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/EC_train_FT5_VQ16.jsonl' --level 16 

CUDA_VISIBLE_DEVICES=1 python foldtoken5/extract_vq_ids.py --path_in  '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/test' --save_vqid_path '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/EC_test_FT5_VQ20.jsonl' --level 20 

CUDA_VISIBLE_DEVICES=2 python foldtoken5/extract_vq_ids.py --path_in  '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/valid' --save_vqid_path '/huyuqi/xmyu/FoldMLM/FoldMLM/datasets/EnzymeCommission/EC_valid_FT5_VQ20.jsonl' --level 20 