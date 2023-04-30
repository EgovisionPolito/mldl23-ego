import models
import torch.nn as nn
from models.I3D import I3D
from models.I3D import InceptionI3d
    
class BaselineTA3N(nn.Module):

    VALID_ENDPOINTS = (
        'i3d',
        'Spatial module',
        'Temporal module',
        'Logits', #Gy
        'Predictions',
    )

    def __init__(self, num_classes=400, final_endpoint='Logits', name='inception_i3d',
                 in_channels=3, model_config=None, backbone='i3d'):
        
        self.end_points = {}
        end_point = 'i3d'
        channel = 3
        backbone = InceptionI3d(num_classes=self.num_class,
                                    in_channels=channel,
                                    model_config=self.model_config)
        weights = I3D.load(self.model_config.weight_i3d_rgb)
        backbone.load_state_dict(weights, strict=False)
        in_features = backbone.logits.in_features
        """
        this is a way to get the number of features at input
        it is the number of features in input before the logits endpoint in I3D
        """
        self.end_points[end_point] = backbone
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Spatial module' # just a fully connected layer
        self.end_points[end_point] = self.SpatialModule(in_features)
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Temporal module'
        self.end_points[end_point] = self.TemporalModule()
        if self._final_endpoint == end_point:
            return

        #missing the final fully connected layer

        pass

    class SpatialModule(nn.Module):
        def __init__(self, in_features, out_features = 1024, model_config=None):
            super(BaselineTA3N.SpatialModule, self).__init__()
            self.in_features = in_features
            self.out_features = min(out_features, in_features)
            """Here I am doing what is done in the official code, 
            in the first fc layer the output dimension is the minimum between the input feature dimension and 1024"""
            self.relu = nn.ReLU(inplace=True) # Again using the architecture of the official code
            self.droput = nn.Dropout(p=model_config.dropout)
            self.fc = nn.Linear(self.in_features, self.out_features)
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x


    class TemporalModule(nn.Module):
        def __init__(self, model_config=None) -> None:
            super(BaselineTA3N.TemporalModule, self).__init__()
            self.pooling = None
            if model_config.temporal_pooling == 'TemPooling':
                pass
            elif model_config.temporal_pooling == 'TemRelation':
                pass
            else:
                raise NotImplementedError
        
        def forward(self, x):
            return self.pooling(x)
    
