import sys

sys.path.insert(0, "../")

import torch
import torchvision
from torch import nn
from thop import profile

feat_dim_of_res50_block = {
    'base' : 64,
    'conv_2' : 256,
    'conv_3' : 512,
    'conv_4' : 1024,
    'conv_5' : 2048
}

prior_dict = {
    "efficientnet-b0": (0.39, 5.3),
    "efficientnet-b1": (0.70, 7.8),
    "efficientnet-b2": (1.00, 9.2),
    "efficientnet-b3": (1.80, 12),
    "efficientnet-b4": (4.20, 19),
    "efficientnet-b5": (9.90, 30),
}

class SqueezeTwice(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)
    
class LstmFc(torch.nn.Module):
    def __init__(self, _feat_dim, _hidden_dim):
        super(LstmFc, self).__init__()
        self.feat_dim = _feat_dim
        self.hidden_dim = _hidden_dim
        
        
    def init_hidden(self, batch_size, cell_size):
        init_cell = torch.Tensor(batch_size, cell_size).zero_()
       
        return init_cell
        
    def forward(self, x):
        hx = self.init_hidden(1, self.hidden_dim)
        cx = self.init_hidden(1, self.hidden_dim)
        hx, cx = torch.nn.LSTMCell(input_size=self.feat_dim, hidden_size=self.hidden_dim)(x,(hx, cx))
        return torch.nn.Linear(self.hidden_dim, 2)(hx)
    

        

def get_gflops_params(model_name, block, num_classes, resolution=224,case=None, hidden_dim=None, seg_len=-1, pretrained=True):
    
    flops = 0
    params = 0
    if model_name in prior_dict:
        gflops, params = prior_dict[model_name]
        gflops = gflops / 224 / 224 * resolution * resolution
        return gflops, params
    
    if "resnet" in model_name:
        base_model = getattr(torchvision.models, model_name)(pretrained)
        
    feat_dim = feat_dim_of_res50_block[block] if block in feat_dim_of_res50_block else getattr(base_model, 'fc').in_features

    if case is "cnn":
        if block is "base":
            model = torch.nn.Sequential(*(list(base_model.children())[:4]))
        elif block is "conv_2":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:4]))
            model = torch.nn.Sequential(*(list(base_model.children())[:5]))
        elif block is "conv_3":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:5]))
            model = torch.nn.Sequential(*(list(base_model.children())[:6]))
        elif block is "conv_4":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:6]))
            model = torch.nn.Sequential(*(list(base_model.children())[:7]))
        elif block is "conv_5":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:7]))
            model = torch.nn.Sequential(*(list(base_model.children())[:8]))
        elif block is "base_fc":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:8]))
            model = torch.nn.Sequential(*(list(base_model.children())[:9]),
                                        SqueezeTwice(),
                                        torch.nn.Linear(feat_dim, num_classes))

        if seg_len == -1:
            dummy_data = torch.randn(1, 3, resolution, resolution)
        else:
            dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)

        hooks = {}
        if block is "base":
            flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
        else:
            sub_flops, sub_params = profile(submodel, inputs=(dummy_data,), custom_ops=hooks)
            flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
            flops  = flops - sub_flops
            params = params - sub_params
        
       
    
    elif case is "rnn" and block is not "base_fc":
        model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            SqueezeTwice(),
            torch.nn.Linear(feat_dim, feat_dim),
            LstmFc(feat_dim, hidden_dim)
        )
        if block is "base":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:4]))
        elif block is "conv_2":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:5]))
        elif block is "conv_3":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:6]))
        elif block is "conv_4":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:7]))
        elif block is "conv_5":
            submodel = torch.nn.Sequential(*(list(base_model.children())[:8]))
            
        
        if seg_len == -1:
            dummy_data = torch.randn(1, 3, resolution, resolution)
        else:
            dummy_data = torch.randn(1, 3, seg_len, resolution, resolution)

        inputs = submodel(dummy_data)
        hooks = {}
        flops, params = profile(model, (inputs,), custom_ops=hooks)

        
        
    gflops = flops / 1e9
    params = params / 1e6   
    
    return gflops, params


if __name__ == "__main__":
    model_name = "resnet50"
    block_list = ["base", "conv_2", "conv_3", "conv_4", "conv_5", "base_fc"]
    num_classes = 100
    case_list = ["cnn", "rnn"]
    hidden_dim=512
    
    for case in case_list:
        for block in block_list:
            _flops, _params = get_gflops_params(model_name, block, num_classes, resolution=192, case=case, hidden_dim=hidden_dim)
            print("%s , %s | %.5f | %.5f" % (case, block, _flops, _params))

    

    
    