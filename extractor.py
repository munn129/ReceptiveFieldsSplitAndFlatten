import sys
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('./cosplace')
from cosplace.cosplace_model import cosplace_network, layers
sys.path.append('./mixvpr')
from mixvpr.main import VPRModel

WEIGHTS = {
    'cosplace' : 'COSPLACE PRE-TRAINED FILE DIR',
    'mixvpr' : 'MIXVPR PRE-TRAINED FILE DIR'
}


class Extractor:
    def __init__(self, loader: DataLoader, dim = 512):

        if not torch.cuda.is_available():
            raise Exception('CUDA must need')
        
        self.device = torch.device('cuda')

        self.loader = loader

        # mixvpr for MLP-MIX
        mixvpr_model = VPRModel(backbone_arch='resnet50',
            layers_to_crop=[4],
            agg_arch='MixVPR',
            agg_config={
                'in_channels' : 1024,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : 256,
                'mix_depth' : 4,
                'mlp_ratio': 1,
                'out_rows' : 2, # the output dim will be (out_rows * out_channels)
                'is_mix' : True
            })
        
        mixvpr_state_dict = torch.load(WEIGHTS['mixvpr'])
        mixvpr_model.load_state_dict(mixvpr_state_dict)
        mixvpr_model = mixvpr_model.to(self.device)
        mixvpr_model.eval()
        self.mixvpr_model = mixvpr_model

        # cosplace for GeM
        cos_model = cosplace_network.GeoLocalizationNet('ResNet152', dim)
        cos_model_state_dict = torch.load(WEIGHTS['cosplace'])
        cos_model.load_state_dict(cos_model_state_dict)
        cos_model = cos_model.to(self.device)
        cos_model.eval()
        self.cos_model = cos_model

        self.matrix = np.empty((loader.__len__(), dim))

    def receptive_field_split_and_flatten(self, image_tensor):
        # cross concatenation
        
        with torch.no_grad():
            mixed_tensor = self.mixvpr_model(image_tensor.to(self.device))
            # 1, 1024, 400

            mixed_tensor = mixed_tensor.view(1,1024,20,20)

            TH = int(mixed_tensor.shape[2]/2)

            top_left = mixed_tensor[:,:,:TH,:TH].view(1, 512, 2, TH, TH).sum(dim=2)
            top_right = mixed_tensor[:,:,:TH, TH:].view(1, 512, 2, TH, TH).sum(dim=2)
            bottom_left = mixed_tensor[:,:,TH:, :TH].view(1, 512, 2, TH, TH).sum(dim=2)
            bottom_right = mixed_tensor[:,:,TH:,TH:].view(1, 512, 2, TH, TH).sum(dim=2)
            # 1, 1024, 10, 10 -> 1, 512, 10, 10

            combined_mix = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)
            # 1, 2048, 10, 10

            des = self.cos_model(combined_mix.to(self.device))

        torch.cuda.empty_cache()

        return des.detach().cpu().numpy()
    
    def feature_extract(self):
        
        for image_tensor, indices in tqdm(self.loader):

            indices_np = indices.detach().numpy()
            
            # self.z_normal(image_tensor)
            # self.z_normalized_mask = np.ones((400,1))
            # self.local_vlad(image_tensor)

            self.matrix[indices_np, :] = self.receptive_field_split_and_flatten(image_tensor)

    def get_matrix(self):
        return self.matrix