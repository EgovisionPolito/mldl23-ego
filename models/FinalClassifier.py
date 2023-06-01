import torch
from torch import nn

from TRNmodule import RelationModuleMultiScale


class Classifier(nn.Module):
    def __init__(self, num_classes, model_args):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.num_clips = model_args.num_clips
        self.avg_modality = model_args.avg_modality
        self.num_classes = num_classes

        self.TRN = RelationModuleMultiScale(1024, 1024, self.num_clips)
        self.TPool = nn.AdaptiveAvgPool2d((1, 1024))

        # define two fully connected layers
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_classes),
            # we use num_classes because loss funct will check how many classes it gets right
            nn.ReLU()
        )

        self.gy = nn.Sequential(
            nn.Linear(1024, self.num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=1):
        x = self.gsf(x)
        # spatial domain out
        # spatial_domain_out = ReverseLayerF.apply(self.gsd(x), alpha)
        # temporal aggregation
        # if (self.temporal_type == "TRN"):
        #     raise NotImplementedError
        # else:
        temporal_aggregation = torch.mean(x, 1)
        # temporal domain
        # temporal_domain_out = ReverseLayerF.apply(self.gtd(temporal_aggregation), alpha)
        class_out = self.gy(temporal_aggregation)

        # return  spatial_domain_out, temporal_domain_out, class_out

        return class_out
