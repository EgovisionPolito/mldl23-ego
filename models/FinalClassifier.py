from torch import nn
import torch
from math import ceil
from models import I3D
from torch.autograd import Function

class Classifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        self.avg_modality = model_args.avg_modality
        self.num_classes = model_args.num_classes
        self.num_clips = model_args.num_clips
        self.baseline_type = model_args.baseline_type
        self.beta = model_args.beta

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 512))

        self.GSF = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.ReLU())

        self.GSD = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.ReLU(),
            nn.Softmax()
            )

        self.fc_classifier_frame = nn.Sequential(
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            nn.Softmax()
        )
        self.fc_classifier_video = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.ReLU(),
            # nn.Softmax()
        )
    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta)
        pred_fc_domain_frame = self.GSD(feat_fc_domain_frame)

        return pred_fc_domain_frame



    def temporal_aggregation(self, x):

        x = x.view(-1, self.num_clips, 512)  # restore the original shape of the tensor
        if self.avg_modality == 'Pooling':
            x = self.AvgPool(x)
        #  self.AvgPool = nn.AdaptiveAvgPool2d((1, 512))
        elif self.avg_modality == 'TRN':
            #x = self.TRN(x)
            pass #per ora serve per l'indentazione
        return x
    def forward(self, input_source, input_target):
        beta = self.beta
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]
        num_segments = self.num_clips
        num_class = self.num_classes

        feat_all_source = []
        feat_all_target = []
        pred_domain_all_source = []
        pred_domain_all_target = []

        # Here we reshape dimensions as batch x num_clip x feat_dim --> (batch * num_clip) x feat_dim because we
        # want to process every clip independently not as part of a batch of clips that refers to the same video because
        # we are at a frame level#
        feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024
        feat_base_target = input_target.view(-1, input_target.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024

        #Adaptive Batch Normalization and shared layers to ask, at the moment we put:
        # Qua questione shared layer non si capisce
        feat_fc_source = self.GSF(feat_base_source) # 160 x 1024 --> 160 x 512
        feat_fc_target = self.GSF(feat_base_target) # 160 x 1024 --> 160 x 512
        #feat_fc_source = feat_base_source
        #feat_fc_target = feat_base_target

        # === adversarial branch (frame-level), in our case clip-level ===#
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta) # 160 x 512 --> 160 x 2
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta) # 160 x 512 --> 160 x 2

    # Da capire un attimo le dimensioni di questo append !!!
        pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
        pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

    #=== prediction (frame-level) ===#
        pred_fc_source = self.fc_classifier_frame(feat_fc_source) # 160 x 512 --> 160 x num_classes
        pred_fc_target = self.fc_classifier_frame(feat_fc_target) # 160 x 512 --> 160 x num_classes

        ### aggregate the frame-based features to relation-based features ###
        feat_fc_video_relation_source = self.temporal_aggregation(feat_fc_source)
        feat_fc_video_relation_target = self.temporal_aggregation(feat_fc_target)

        feat_fc_video_source = feat_fc_video_relation_source.sum(1)  # 32 x 4 x 512 --> 32 x 512
        feat_fc_video_target = feat_fc_video_relation_target.sum(1)  # 32 x 4 x 512 --> 32 x 512

        pred_fc_video_source = self.fc_classifier_video(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video(feat_fc_video_target)

        #=== final output ===# inutile
        #output_source = self.final_output(pred_fc_source, pred_fc_video_source, num_segments)
        #output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

        results = {
            'domain_source':pred_domain_all_source ,
            'domain_target': pred_domain_all_target,
            'pred_frame_source': pred_fc_source,
            'pred_frame_target': pred_fc_target,
            'pred_video_target': pred_fc_video_target,
            'pred_video_source': pred_fc_video_source,
        }

        return results, {}
        #return  pred_fc_source, pred_fc_video_source, pred_domain_all_source, pred_fc_target, pred_fc_video_target , pred_domain_all_target

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None
