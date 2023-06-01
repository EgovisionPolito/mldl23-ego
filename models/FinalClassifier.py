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

        self.gsf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            # we use num_classes because loss funct will check how many classes it gets right
            nn.Linear(1024, self.num_classes),
            nn.ReLU()
        )

        self.g_y = nn.Sequential(
            nn.Linear(1024, self.num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.gsf(x)

        # temporal aggregation
        temporal_aggregation = torch.mean(x, 1)
        output = self.g_y(temporal_aggregation)

        return output
