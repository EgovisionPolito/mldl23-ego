import torch
from torch import nn
from torch.autograd import Function


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
        self.beta = model_args.beta
        self.TRN = RelationModuleMultiScale(1024, 1024, self.num_clips)
        self.TPool = nn.AdaptiveAvgPool2d((1, 1024))

        # the input will be 5x1024 for features_rgb inside the pickles
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
            nn.Linear(512, 2),  # TODO why 2?
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
        self.relation_domain_classifier_all = nn.ModuleList()
        for i in range(self.num_clips - 1):
            relation_domain_classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
            )
            self.relation_domain_classifier_all += [relation_domain_classifier]

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

    # same as UDA, this would call the GRD function previously defined
    def domain_classifier_relation(self, feat_relation, beta):

        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):  # HERE we call the GRD
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
        pred_fc_domain_video = self.GTD(feat_fc_domain_video)

        return pred_fc_domain_video

    # method to either pool or TRN
    def temporal_modality(self, x):

        x = x.view(-1, self.num_clips, 1024)  # view is used to change the dimension, restoring the original shape

        if self.avg_modality == 'Pooling':
            x = self.TPool(x)
            # x = x.view(-1, 1024)
        elif self.avg_modality == 'TRN':
            x = self.TRN(x)
        return x

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
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target,
                                                                   beta)  # 160 x 1024 --> 160 x 2, prediction domain output

        # same as UDA
        # pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
        # pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

        pred_fc_domain_frame_source = pred_fc_domain_frame_source.view(
            (batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:])
        pred_fc_domain_frame_target = pred_fc_domain_frame_target.view(
            (batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:])

        # changed the method to be the same of our code, which calls GSD
        # === source layers (frame-level) ===#
        pred_fc_source = self.fc_classifier_frame(feat_fc_source)  # 160 x 1024 --> 160 x num_classes
        pred_fc_target = self.fc_classifier_frame(feat_fc_target)  # 160 x 1024 --> 160 x num_classes

        ### aggregate the frame-based features to relation-based features, temporal modality automatically chooses the right one ###
        feat_fc_video_relation_source = self.temporal_modality(feat_fc_source)
        feat_fc_video_relation_target = self.temporal_modality(feat_fc_target)

        # if we apply AvgPool we will only have 1x1024, so no need to apply GRD because it is as if we were already
        # at video level since we already have only one "summary feature" per video (not conceptually right)
        if self.avg_modality == 'TRN':  # we have 4 frames relations
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source,
                                                                                   beta)  # 32 x 4 x 1024 --> 32 x 2
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target,
                                                                                   beta)  # 32 x 4 x 1024 --> 32 x 2

        # === prediction (video-level) ===#
        # sum up relation features (ignore 1-relation)
        # in case of AVGpool we already only have 1 clip so it would be pointless to sum, but we leave it like this for
        # better clarity
        feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)  # 32 x 4 x 1024 --> 32 x 1024
        feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)  # 32 x 4 x 1024 --> 32 x 1024

        # now we have to make the classifier on both the previously found things
        pred_fc_video_source = self.fc_classifier_video(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video(feat_fc_video_target)

        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)

        # now we aggregate all the domain prediction in one single variable
        if self.avg_modality == 'TRN':  # append domain_relation_predictions
            num_relation = feat_fc_video_relation_source.size()[1]
            # pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
            # pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))

            pred_fc_domain_video_relation_source = pred_fc_domain_video_relation_source.view(
                (batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:])
            pred_fc_domain_video_relation_target = pred_fc_domain_video_relation_target.view(
                (batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:])
        elif self.avg_modality == 'Pooling':
            # pred_domain_all_source.append(pred_fc_domain_video_source)  # if not trn-m, add dummy tensors for relation features
            # pred_domain_all_target.append(pred_fc_domain_video_target)  # same as above
            pred_fc_domain_video_relation_source = pred_fc_domain_video_source
            pred_fc_domain_video_relation_target = pred_fc_domain_video_target

        # instead of using final output we directly aggregate here
        # pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
        # pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))
        pred_fc_domain_video_source = pred_fc_domain_video_source.view(
            (batch_source,) + pred_fc_domain_video_source.size()[-1:])
        pred_fc_domain_video_target = pred_fc_domain_video_target.view(
            (batch_target,) + pred_fc_domain_video_target.size()[-1:])

        # results = {
        #     'domain_source': pred_domain_all_source,
        #     'domain_target': pred_domain_all_target,
        #     'pred_frame_source': pred_fc_source,
        #     'pred_video_source': pred_fc_video_source,
        #     'pred_frame_target': pred_fc_target,
        #     'pred_video_target': pred_fc_video_target,
        # }

        results = {
            'domain_source_frame': pred_fc_domain_frame_source,
            'domain_target_frame': pred_fc_domain_frame_target,
            'domain_source_relation': pred_fc_domain_video_relation_source,
            'domain_target_relation': pred_fc_domain_video_relation_target,
            'domain_source_video': pred_fc_domain_video_source,
            'domain_target_video': pred_fc_domain_video_target,
            'pred_frame_source': pred_fc_source,
            'pred_video_source': pred_fc_video_source,
            'pred_frame_target': pred_fc_target,
            'pred_video_target': pred_fc_video_target,
        }
        # return [pred_domain_all_source, pred_domain_all_target, pred_fc_source, pred_fc_video_source, pred_fc_target, pred_fc_video_target], {}
        return results, {}
    # def forward(self, x):
    #     x = self.gsf(x)
    #
    #     # temporal aggregation
    #     temporal_aggregation = torch.mean(x, 1)
    #     output = self.g_y(temporal_aggregation)
    #
    #     return output


class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3  # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)]  # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num,
                                             len(relations_scale)))  # how many samples of relation to select in each forward pass

        # self.num_class = num_class
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList()  # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
            )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_scale_1 = input[:, self.relations_scales[0][0], :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1)  # add one dimension for the later concatenation
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in
                                        range(num_select_relations)]

            # for idx in idx_relations_randomsample:
            for idx in idx_relations_evensample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = act_relation.unsqueeze(1)  # add one dimension for the later concatenation
                act_relation_all += act_relation

            act_all = torch.cat((act_all, act_relation_all), 1)
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
