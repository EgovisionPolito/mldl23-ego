from torch import nn
from torch.autograd import Function
import torch

class BaseClassifier(nn.Module):
    def __init__(self, num_class, n_features,temporal_type,ablation_mask):
        super().__init__()
        """
        n_features: [0]: 5
                    [1]: 1024
        tmeporal_type: TRN or pooling
        ablation_mask: [gsd,gtd,grd,domain_att]

        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.num_class = num_class
        self.n_feat = n_features
        self.temporal_type = temporal_type
        self.ablation_mask = ablation_mask
        
        #GSF
        n_gsf_out = 512
        self.gsf = nn.Sequential()
        self.gsf.add_module('gsf_fc1', nn.Linear(self.n_feat[1], n_gsf_out))
        self.gsf.add_module('gsf_bn1', nn.BatchNorm1d(n_features[0]))
        self.gsf.add_module('gsf_relu1', nn.ReLU(True))
        self.gsf.add_module('gsf_drop1', nn.Dropout())
        self.gsf.add_module('gsf_fc2', nn.Linear(n_gsf_out, n_gsf_out))
        self.gsf.add_module('gsf_bn2', nn.BatchNorm1d(n_features[0]))
        self.gsf.add_module('gsf_relu2', nn.ReLU(True))

        #Gy
        self.gy = nn.Sequential()
        self.gy.add_module('c_fc1', nn.Linear(n_gsf_out, num_class))
        self.gy.add_module('c_softmax', nn.LogSoftmax(dim=1))

        " Domain adaptation **********************************************************************"
        #gtd 
        self.gtd = nn.Sequential()
        self.gtd.add_module('gtd_fc1', nn.Linear(n_gsf_out, 512))
        self.gtd.add_module('gtd_bn1', nn.BatchNorm1d(512))
        self.gtd.add_module('gtd_relu1', nn.ReLU(True))
        self.gtd.add_module('gtd_fc3', nn.Linear(512, 2))
        self.gtd.add_module('gtd_softmax', nn.LogSoftmax(dim=1))
 
    def forward(self, x,alpha = 1):
        gsf_out = self.gsf(x)
        if(self.temporal_type == "TRN"):
            raise NotImplementedError
        else:
            gtf_out = torch.mean(gsf_out,1)
        
        if(self.ablation_mask[1]):
            temp_dom = self.gtd(ReverseLayerF.apply(gtf_out,1))
        
        class_logit = self.gy(gtf_out)
        
        return class_logit,temp_dom
        
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
