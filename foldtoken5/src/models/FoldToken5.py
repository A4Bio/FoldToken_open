import torch
import torch.nn as nn
from src.modules.FoldToken5_module import *
from src.chroma.data import Protein
from src.modules.vq_modules5 import  FSQ
from torch_geometric.nn import knn_graph
# from src.chroma.layers.structure.rmsd import CrossRMSD_Graph



class PiFold_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(PiFold_Model, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        self.encoder = StructureEncoder(3, 3, 3, 3, args.enc_layers, hidden_dim, dropout=0.0)
        self.vq = FSQ(args.levels, hidden_dim, vq_dim=len(args.levels))
        self.decoder = StructDecoder(args.dec_layers, hidden_dim)
        # self.rmsd = CrossRMSD_Graph()
        self._init_params()
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    

    def forward(self, batch,  vq_only=False):
        V, E, T, T_ts, batch_id_extend, edge_idx_extend, chain_encoding = GeoFeaturizer.from_X_to_features(batch['X'], batch['idx'], batch['edge_idx'], batch['batch_id'], batch['chain_encoding'], virtual_frame_num=0, input_layer=True)

        X, edge_idx, batch_id = batch['X'], batch['edge_idx'], batch['batch_id']
        
        h_V, h_E, T_ts, edge_idx, batch_id = self.encoder.forward_stage1(batch['X'], batch['idx'], edge_idx, batch_id, chain_encoding, V, E, T_ts, batch_id_extend, edge_idx_extend, virtual_frame_num=0)
        
        h_V = self.encoder(h_V, h_E, T_ts, edge_idx, batch_id)
        h_V, vq_code = self.vq(h_V)    
        if vq_only:
            return h_V, vq_code
        
        all_preds = self.decoder(h_V, batch['idx'], batch_id, chain_encoding, dec_topk=self.args.dec_topk)
            
        # X_pred = all_preds[-1]
        # X = X_pred[batch_id==1][None]
        # C = S = torch.ones_like(X)[:,:,0,0].long()
        # protein = Protein.from_XCS(X,C,S)
        
        
        return  all_preds, h_V, vq_code
            


    def sample(self, 
               protein_init):
        self.eval()
        X, C, S = protein_init.to_XCS()
        X = X[C>0][None,]
        S = S[C>0][None,]
        C = C[C>0][None,]
        
        batch_id = torch.zeros_like(S)[0]
        X = X[0]
        edge_idx = knn_graph(X[:,1], k=30, batch=batch_id, loop=True, flow='target_to_source')
        chain_encoding = torch.zeros_like(batch_id)


        V, E, T, T_ts, batch_id_extend, edge_idx_extend, chain_encoding = GeoFeaturizer.from_X_to_features(X, edge_idx, batch_id, chain_encoding, virtual_frame_num=3)
        X_t = torch.rand_like(X)
        temp = 1e-8
        mode = 'eval'

        X_pred, all_preds, vq_loss, vq_count = self(X, chain_encoding,
                                    edge_idx, batch_id, 
                                    X_t, V, E, T_ts, 
                                    temp, mode, 
                                    batch_id_extend = batch_id_extend,
                                    edge_idx_extend = edge_idx_extend,
                                    virtual_frame_num=3,
                                    vqshortcut = 0,
                                    frozen_vq=0)
        
        # X_pred, vq_ids = self(**batch, return_predX=True)
        X_pred = X_pred[None,...]
        protein = Protein.from_XCS(X_pred, C, S)

        return protein

