import inspect
import torch
import torch.nn as nn
from src.interface.model_interface import MInterface_base
from src.modules.FoldToken5_module import GeoFeaturizer
from torch_geometric.nn import knn_graph
from src.chroma.data import Protein
from src.tools.geo_utils import batch_rmsd

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = nn.NLLLoss(reduction='none')

    def forward(self, batch, vq_only=False):
        all_preds, h_V, vq_code = self.model(batch, vq_only=vq_only)
        return {'all_preds': all_preds, 'h_V':h_V, 'vq_code': vq_code} 

    
    def load_model(self):
        from src.models.FoldToken5 import PiFold_Model
        self.model = PiFold_Model(self.hparams)


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
    
    # def batch_data_single_chain(self, list_of_data):
    #     name_all = []
    #     batch_id_all = []
    #     chain_encoding_all = []
    #     X_all = []
    #     S_all = []
    #     node_idx_all = []
    #     idx = 0
    #     for (name, X, C, S) in list_of_data:
    #         for j in C.unique():
    #             mask = C == j
    #             if mask.float().sum()<30:
    #                 continue
    #             batch_id = torch.zeros_like(S[mask])+idx
    #             chain_encoding = C[mask]
    #             name_all.append(name+'_'+str(j.item()))
    #             batch_id_all.append(batch_id)
    #             chain_encoding_all.append(chain_encoding)
    #             X_all.append(X[mask])
    #             S_all.append(S[mask])
    #             node_idx_all.append(torch.arange(batch_id.shape[0], device=batch_id.device))
    #             idx += 1
        
    #     S_all = torch.cat(S_all, dim=0)
    #     X_all = torch.cat(X_all, dim=0)
    #     batch_id_all = torch.cat(batch_id_all, dim=0)
    #     chain_encoding_all = torch.cat(chain_encoding_all, dim=0)
    #     node_idx_all = torch.cat(node_idx_all, dim=0)
    #     edge_idx = knn_graph(X_all[:,1], k=50, batch=batch_id_all, loop=True, flow='target_to_source')
    #     return {'title':name_all ,'S':S_all, 'X': X_all, 'edge_idx': edge_idx, 'batch_id': batch_id_all, 'chain_encoding': chain_encoding_all, 'idx':node_idx_all}
    
    def batch_data(self, list_of_data):
        name_all = []
        batch_id_all = []
        chain_encoding_all = []
        X_all = []
        S_all = []
        node_idx_all = []
        idx = 0
        for (name, X, C, S) in list_of_data:
            batch_id = torch.zeros_like(S)+idx
            chain_encoding = C
            name_all.append(name)
            batch_id_all.append(batch_id)
            chain_encoding_all.append(chain_encoding)
            X_all.append(X)
            S_all.append(S)
            node_idx_all.append(torch.arange(batch_id.shape[0], device=batch_id.device))
            idx += 1
        
        S_all = torch.cat(S_all, dim=0)
        X_all = torch.cat(X_all, dim=0)
        batch_id_all = torch.cat(batch_id_all, dim=0)
        chain_encoding_all = torch.cat(chain_encoding_all, dim=0)
        node_idx_all = torch.cat(node_idx_all, dim=0)
        edge_idx = knn_graph(X_all[:,1], k=50, batch=batch_id_all, loop=True, flow='target_to_source')
        return {'title':name_all ,'S':S_all, 'X': X_all, 'edge_idx': edge_idx, 'batch_id': batch_id_all, 'chain_encoding': chain_encoding_all, 'idx':node_idx_all}
    
    def filter_nan_data(self, X, C, S):
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X,(-2,-1))).float()
        X[isnan] = 0.
        
        mask_bool = (mask==1)
        def node_mask_select(x):
            shape = x.shape
            x = x.reshape(shape[0], -1)
            out = torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
            out = out.reshape(-1,*shape[1:])
            return out
        
        S = node_mask_select(S)
        X = node_mask_select(X)
        C = node_mask_select(C)
        return X, C, S
    
    def batch_proteins(self, list_of_proteins):
        name_all = []
        batch_id_all = []
        chain_encoding_all = []
        X_all = []
        S_all = []
        node_idx_all = []
        for idx, protein in enumerate(list_of_proteins):
            # name = protein.sys.name.split('/')[-1]
            X, C, S = protein.to_XCS()
            X = X[C>0][None,]
            S = S[C>0][None,]
            C = C[C>0][None,]
            L = X.shape[1]
            batch_id = torch.zeros_like(S)[0]+idx
            chain_encoding = C[0]
            # name_all.append(name)
            batch_id_all.append(batch_id)
            node_idx_all.append(torch.arange(batch_id.shape[0], device=batch_id.device))
            chain_encoding_all.append(chain_encoding)
            X_all.append(X[0])
            S_all.append(S[0])
        
        S_all = torch.cat(S_all, dim=0)
        X_all = torch.cat(X_all, dim=0)
        batch_id_all = torch.cat(batch_id_all, dim=0)
        node_idx_all = torch.cat(node_idx_all, dim=0)
        chain_encoding_all = torch.cat(chain_encoding_all, dim=0)
        edge_idx = knn_graph(X_all[:,1], k=50, batch=batch_id_all, loop=True, flow='target_to_source')
        return {'S':S_all, 'X': X_all, 'edge_idx': edge_idx, 'batch_id': batch_id_all, 'chain_encoding': chain_encoding_all, 'temp': 1e-8, 'mode': 'eval', 'idx':node_idx_all}
    

    def sample(self,  batch, cal_rmsd=False):
        self.eval()
        ret = self(batch)
        X_pred = ret['all_preds'][-1]
        batch_id = batch['batch_id']
        
        vq_code_all = []
        protein_all = []
        
        for i in range(batch_id.max()+1):
            X = X_pred[batch_id==i]
            C = batch['chain_encoding'][batch_id==i]
            S = torch.ones_like(C)
            protein = Protein.from_XCS(X[None], C[None], S[None])
            protein_all.append(protein)
            vq_code_all.append(ret['vq_code'][batch_id==i])
        
        if cal_rmsd:
            rmsd = batch_rmsd(X_pred[:,1], batch['X'][:,1], batch_id)
            return protein_all, vq_code_all, rmsd
        return protein_all, vq_code_all
    
    def decode(self, vq_id, node_idx, batch_id, chain_encoding):
        h_V = self.model.vq.indexes_to_codes(vq_id)
        h_V = self.model.vq.proj_inv(h_V)
        all_preds = self.model.decoder(h_V, node_idx, batch_id, chain_encoding, dec_topk=30)
        X_pred = all_preds[-1]
        return X_pred
    
    def encode_protein(self, protein_init, level=8, Ablation=False):
        self.eval()
        X, C, S = protein_init.to_XCS()
        X = X[C>0][None,]
        S = S[C>0][None,]
        C = C[C>0][None,]
        
        batch_id = torch.zeros_like(S)[0]
        X = X[0]
        edge_idx = knn_graph(X[:,1], k=50, batch=batch_id, loop=True, flow='target_to_source')
        chain_encoding = torch.zeros_like(batch_id)+1
        node_idx = torch.arange(batch_id.shape[0], device=batch_id.device)

        batch = {'X': X, 'idx':node_idx, 'edge_idx': edge_idx, 'batch_id': batch_id, 'chain_encoding': chain_encoding}
        
        h_V, vq_code = self.model(batch, vq_only=True)

        return vq_code, batch_id, node_idx, chain_encoding
