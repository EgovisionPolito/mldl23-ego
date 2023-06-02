import torch
from torch import nn
from torch.autograd import Function

from TRNmodule import RelationModuleMultiScale


# definition of Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class Classifier(nn.Module):
    def __init__(self, num_classes, model_args):
        super().__init__()
        self.num_clips = model_args.num_clips
        self.avg_modality = model_args.avg_modality
        self.num_classes = num_classes

        self.TRN = RelationModuleMultiScale(1024, 1024, self.num_clips)
        self.TPool = nn.AdaptiveAvgPool2d((1, 1024))

        #the input will be 5x1024 for features_rgb inside the pickles
        self.gsf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )

        self.g_y = nn.Sequential(
            nn.Linear(1024, self.num_classes),
            nn.LogSoftmax(dim=1)
        )

        # Spatial domain discriminator (GSD)
        self.GSD = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2), # TODO why 2?
            nn.LogSoftmax(dim=1)
        )

        # Temporal domain discriminator (GTD)
        self.GTD = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LogSoftmax(dim=1)
        )

        # Relational domain discriminator (GRD)
        self.GRD = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LogSoftmax(dim=1)
        )

        self.fc_classifier_frame = nn.Sequential(
            nn.Linear(1024, self.num_classes),
            nn.Softmax())

        self.fc_classifier_video = nn.Sequential(
            nn.Linear(1024, self.num_classes),
            nn.Softmax())


    # same as UDA
    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta)
        pred_fc_domain_frame = self.GSD(feat_fc_domain_frame)

        return pred_fc_domain_frame

    # same as UDA
    def domain_classifier_relation(self, feat_relation, beta):
        #
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:, i, :].squeeze(1)  # 32x1x1024 -> 32x1024
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single,
                                                               beta)  # the same beta for all relations (for now)

            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)

            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1, 1, 2)
            else:
                pred_fc_domain_relation_video = torch.cat(
                    (pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1, 1, 2)), 1)

        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1, 2)

        return pred_fc_domain_relation_video

    # same as UDA
    def domain_classifier_video(self, feat, beta):
        feat_fc_domain_video = GradReverse.apply(feat, beta)
        pred_fc_domain_video = self.GVD(feat_fc_domain_video)

        return pred_fc_domain_video


    # method to either pool or TRN
    def temporal_aggregation(self, x):

        x = x.view(-1, self.num_clips, 1024)  # view is used to change the dimension

        if self.avg_modality == 'Pooling':
            x = self.AvgPool(x)
            # x = x.view(-1, 1024)
        elif self.avg_modality == 'TRN':
            x = self.TRN(x)
        return x

    # removed some excessive code from UDA
    def final_output(self, pred, pred_video):
        if self.baseline_type == 'video':
            base_out = pred_video
        else:
            base_out = pred

        output = base_out
        return output

    def forward(self, input_source, input_target):
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]
        num_segments = self.num_clips
        beta = self.beta

        pred_domain_all_source = []
        pred_domain_all_target = []

        # In order to handle each clip separately and not treat them as a batch that belongs to the same video,
        # we modify the dimensions from (batch x num_clip x feat_dim) to ( (batch * num_clip) x feat_dim ).
        # This allows to process each clip independently and focus on individual frames
        feat_base_source = input_source.view(-1, input_source.size()[-1])  # 32 x 5 x 1024 --> 160 x 1024
        feat_base_target = input_target.view(-1, input_target.size()[-1])  # 32 x 5 x 1024 --> 160 x 1024

        # Adaptive BN (batch normalization)
        # put our fc called gsf
        feat_fc_source = self.gsf(feat_base_source)
        feat_fc_target = self.gsf(feat_base_target)
        # now our dimensions are: 160 x 1024

        # === adversarial branch (frame-level), chiara said: clip-level ===#
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta)  # 160 x 1024 --> 160 x 2
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)  # 160 x 1024 --> 160 x 2

        # Da capire un attimo le dimensioni di questo append !!!
        pred_domain_all_source.append(
            pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
        pred_domain_all_target.append(
            pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

        # === prediction (frame-level) ===#
        pred_fc_source = self.fc_classifier_frame(feat_fc_source)  # 160 x 1024 --> 160 x num_classes
        pred_fc_target = self.fc_classifier_frame(feat_fc_target)  # 160 x 1024 --> 160 x num_classes

        ### aggregate the frame-based features to relation-based features ###
        feat_fc_video_relation_source = self.temporal_aggregation(feat_fc_source)
        feat_fc_video_relation_target = self.temporal_aggregation(feat_fc_target)

        # Here if we have used AvgPool we obtain a tensor of size batch x feat_dim, otherwise if we have used TRN we obtain
        # a tensor of size batch x num_relations x feat_dim and we have to implement a domain classifier for the TRN case
        # so i think we need to do something as:
        # if self.avg_modality == 'TRN':
        #  pred_fc_domain_relation_source = self.domain_classifier_relation(feat_fc_relation_source, beta)
        #  pred_fc_domain_relation_target = self.domain_classifier_relation(feat_fc_relation_target, beta)
        # where domain_classifier_relation have to be implemented as fc where num_relation x feat_dim -> 2
        # pay attention that aggregated data with AvgPool are yet at video level (1x1024) so doing a relation domain classifier
        # on them is useful because we will do it later so we have to put
        # elif self.avg_modality == 'Pooling':
        # feat_fc_video_source = feat_fc_relation_source
        # feat_fc_video_target = feat_fc_relation_target

        if self.avg_modality == 'TRN':  # we have 4 frames relations
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source,
                                                                                   beta)  # 32 x 4 x 1024 --> 32 x 2
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target,
                                                                                   beta)  # 32 x 4 x 1024 --> 32 x 2

        # === prediction (video-level) ===#
        # aggregate the frame-based features to video-based features, we can use sum() even in AVGPOOL case because we have
        # alredy only 1 "clip" dimension (batch x feat_dim)

        feat_fc_video_source = feat_fc_video_relation_source.sum(1)  # 32 x 4 x 1024 --> 32 x 1024
        feat_fc_video_target = feat_fc_video_relation_target.sum(1)  # 32 x 4 x 1024 --> 32 x 1024

        pred_fc_video_source = self.fc_classifier_video(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video(feat_fc_video_target)

        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)

        # what does he do in the code: he append the DOMAIN predictions of the frame-level and video-level
        # indipendentemente from the aggregation method, then appends domain_relation_predictions only if we have
        # used TRN as aggregation method or another time the same domain_video_predictions if we have used AVGPOOL as
        # aggregation method

        if self.avg_modality == 'TRN':  # append domain_relation_predictions

            num_relation = feat_fc_video_relation_source.size()[1]
            pred_domain_all_source.append(pred_fc_domain_video_relation_source.view(
                (batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
            pred_domain_all_target.append(pred_fc_domain_video_relation_target.view(
                (batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))

        elif self.avg_modality == 'Pooling':  # append domain_video_predictions again
            pred_domain_all_source.append(
                pred_fc_domain_video_source)  # if not trn-m, add dummy tensors for relation features
            pred_domain_all_target.append(
                pred_fc_domain_video_target)  # if not trn-m, add dummy tensors for relation features

        pred_domain_all_source.append(
            pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
        pred_domain_all_target.append(
            pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

        # === final output ===#
        # output_source = self.final_output(pred_fc_source, pred_fc_video_source, num_segments)
        # output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

        results = {
            'domain_source': pred_domain_all_source,
            'domain_target': pred_domain_all_target,
            'pred_frame_source': pred_fc_source,
            'pred_video_source': pred_fc_video_source,
            'pred_frame_target': pred_fc_target,
            'pred_video_target': pred_fc_video_target,
        }

        return results, {}
        # return  pred_fc_source, pred_fc_video_source, pred_domain_all_source, pred_fc_target, pred_fc_video_target
        # , pred_domain_all_target

    def forward(self, x):
        x = self.gsf(x)

        # temporal aggregation
        temporal_aggregation = torch.mean(x, 1)
        output = self.g_y(temporal_aggregation)

        return output
