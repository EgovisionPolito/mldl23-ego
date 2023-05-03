import models
import torch
import torch.nn as nn
from models.I3D import I3D
from models.I3D import InceptionI3d
from torch.nn.init import normal_, constant_
import TRNmodule
    
class BaselineTA3N(nn.Module):

    VALID_ENDPOINTS = (
        'Backbone',
        'Spatial module',
        'Temporal module',
        'Gy',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, model_config=None, backbone='i3d'):
        
        self.end_points = {}
        self.TRN = TRNmodule.RelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
        end_point = 'Backbone'
        """
        this is a way to get the number of features at input
        it is the number of features in input before the logits endpoint in I3D
        """
        self.end_points[end_point] = self.FeatureExtractorModule(model_config=model_config)
        backbone = self.end_points[end_point]
        feat_dim = backbone.feat_dim
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Spatial module' # just a fully connected layer
        fc_spatial_module = self.FullyConnectedLayer(feat_dim)
        std = 0.001
        constant_(fc_spatial_module.bias, 0)
        normal_(fc_spatial_module.weight, 0, std)
		
        self.end_points[end_point] = fc_spatial_module # spatial module is just a fully connected layer
        
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Temporal module'
        self.end_points[end_point] = self.TemporalModule()
        if self._final_endpoint == end_point:
            return

        end_point = 'Gy'
        fc_gy = self.FullyConnectedLayer(feat_dim)
        constant_(fc_gy.bias, 0)
        normal_(fc_gy.weight, 0, std)

        self.end_points[end_point] = fc_gy
        if self._final_endpoint == end_point:
            return
        #missing the final fully connected layer

        pass

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
        pass

    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features, out_features = 1024, dropout=0.8):
            super(BaselineTA3N.FullyConnectedLayer, self).__init__()
            self.in_features = in_features
            self.out_features = min(out_features, in_features)
            """Here I am doing what is done in the official code, 
            in the first fc layer the output dimension is the minimum between the input feature dimension and 1024"""
            self.relu = nn.ReLU(inplace=True) # Again using the architecture of the official code
            self.droput = nn.Dropout(p=dropout)
            self.fc = nn.Linear(self.in_features, self.out_features)
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x


    class TemporalModule(nn.Module):
        def __init__(self, in_features_dim, temporal_pooling = 'TemPooling') -> None:
            super(BaselineTA3N.TemporalModule, self).__init__()
            self.pooling = None
            self.in_features_dim = in_features_dim
            if temporal_pooling == 'TemPooling':
                pass
            elif temporal_pooling == 'TemRelation':
                self.num_bottleneck = 512
                self.pooling = TRNmodule.RelationModule(in_features_dim, self.num_bottleneck, self.train_segments)
                self.out_features_dim = self.num_bottleneck
                pass
            else:
                raise NotImplementedError
        
        def forward(self, x):
            return self.pooling(x)
    
    class FeatureExtractorModule(nn.Module):

        VALID_BACKBONES = {
            'i3d': I3D
        }

        def __init__(self, num_class, modality, model_config, **kwargs):
            super(BaselineTA3N.FeatureExtractorModule, self).__init__()
            self.backbone = I3D(num_class, modality, model_config, **kwargs)
            self.feat_dim = self.backbone.feat_dim
        
        def forward(self, x):
            logits, features = self.backbone(x)
            features = features['feat']
            return features.view(-1, features.size()[-1]) 

