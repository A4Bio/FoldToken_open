# Open-source version of FoldToken4

Creating protein structure language has attracted increasing attention in unifing the modality of protein sequence and structure. While recent works, such as FoldToken1&2&3 have made great progress in this direction, the relationship between languages created by different models at different scales is still unclear. We propose FoldToken4 to learn the consistency and hierarchy of multiscale fold languages, and share the code here to promote the research process.

## Demo Video

## Usage
```bash
export PYTHONPATH=project_path

CUDA_VISIBLE_DEVICES=0 python reconstruct.py --path_in ./N128 --path_out ./N128_pred --level 5
```
 One can use following script to validate the reconstruction performance of FoldToken3. The molel will encode input pdbs in **`path_in`**, reconstruct them, and save the reconstructed structures to  **`path_out`**. Users can specify **`config`** and **`checkpoint`** to select appropriate model. The codebook size is $2^{level}$, i.e., $2^{5}$ in the example.

```bash
export PYTHONPATH=project_path

CUDA_VISIBLE_DEVICES=0 python extract_vq_ids.py --path_in ./N128 --save_vqid_path ./N128_vqid.jsonl --level 8
```
 One can use following script to extract vq ids from pdbs in **`path_in`**, and save it to **`path_out`**. Users can specify **`config`** and **`checkpoint`** to select appropriate model. The codebook size is $2^{level}$, i.e., $2^{8}$ in the example.

## Why we do this?
### What's the ultimate goal?
Our ultimate goal is to unify multi-modality molecule modeling technique. The unification comes from:
1. Data Unification: unify any modality data to a sigle modality (sequence representation).
2. Model Unification: using the unified model architecture as much as possible.
This unification will greatly simplify molecular modeling and change the underlying modeling techniques. Finally, we can build a unified bio-language that will collaborate with human languages to facilitate molecular discovery.

### Why we start from structure language?
**1. Efficiency:** Over our research, we find that the computing cost will be significant and the underlying technique is complex when we address structure-related problems. If we can transform structures as a foreign language, things will change: researchers with few computing sources can join the game of structure modeling. This is the practical value of fold language.

**2. New Technique Route:** Scientifically speaking, we can open up a new technical route to reforming structure-related tasks. This is an orthogonal research direction to current structure-diffusion models. In image modeling, researchers have proven that language model can beat diffusion models with a good tokenizer. With continual efforts, we believe this success will be apparent in molecule structure modeling.

## Welcome suggestions.

Welcome good suggestions about this project. Please contact: gaozhangyang@westlake.edu.cn.


<!-- I'm a AI scientists studying bio-problems, my CV is -->



<!-- Welcome Bio researchers to suggest important bio-problems. -->
