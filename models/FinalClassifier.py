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
        self.beta = model_args.beta

        self.AvgPool = nn.AdaptiveAvgPool2d((1, 512))
        self.TRN = RelationModuleMultiScale(512,512,self.num_clips)

        self.domain_adapt_strategy = model_args.domain_adapt_strategy
        self.use_attn = model_args.use_attn

        self.GSF = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
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
        self.GTD = nn.Sequential( #questo si dovrebbe chiamare GTD
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.ReLU(),
            nn.Softmax()
        )

        # domain classifier for TRN-M (nel nostro caso M = 4)
        if self.avg_modality == 'TRN':
            self.relation_domain_classifier_all = nn.ModuleList()
            for i in range(self.num_clips - 1):
                relation_domain_classifier = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 2),
                    nn.Softmax() # ci potrebbe stare così abbiamo delle probabilità di appartenenza ad un determinato dominio
                )
                self.relation_domain_classifier_all += [relation_domain_classifier]


        self.fc_classifier_frame = nn.Sequential(
            nn.Linear(512, self.num_classes),
            nn.ReLU()
        )
        self.fc_classifier_video = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
            nn.ReLU()
        )

    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta)
        pred_fc_domain_frame = self.GSD(feat_fc_domain_frame)

        return pred_fc_domain_frame
    def domain_classifier_video(self, feat, beta):
        feat_fc_domain_video = GradReverse.apply(feat, beta)
        pred_fc_domain_video = self.GTD(feat_fc_domain_video)

        return pred_fc_domain_video

    def domain_classifier_relation(self, feat_relation, beta): #copiato da UDA TA3N
        # 32x4x512 --> (32x4)x2
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:, i, :].squeeze(1)  # 32x1x512 -> 32x512
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single,beta)  # the same beta for all relations (for now)
            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)

            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1, 1, 2)
            else:
                pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1, 1, 2)), 1)

        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1, 2)

        return pred_fc_domain_relation_video #output size : [128,2]

    def temporal_aggregation(self, x):

        x = x.view(-1, self.num_clips, 512)  # restore the original shape of the tensor, cioè 32x5x512
        if self.avg_modality == 'Pooling':
            x = self.AvgPool(x)   # 32x5x512 -> 32x1x512
        #  self.AvgPool = nn.AdaptiveAvgPool2d((1, 512))
        elif self.avg_modality == 'TRN':
            x = self.TRN(x)
        return x

    def get_trans_attn(self, pred_domain):
        #softmax = nn.Softmax(dim=1)
        #logsoftmax = nn.LogSoftmax(dim=1)
        #entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        entropy = torch.sum(-(pred_domain) * torch.log(pred_domain), 1)
        weights = 1 - entropy

        return weights

    def get_general_attn(self, feat):
        num_segments = feat.size()[1]
        feat = feat.view(-1, feat.size()[-1]) # reshape features: 128x4x256 --> (128x4)x256
        weights = self.attn_layer(feat) # e.g. (128x4)x1
        weights = weights.view(-1, num_segments, weights.size()[-1]) # reshape attention weights: (128x4)x1 --> 128x4x1
        weights = nn.Softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

        return weights

    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        if self.use_attn == 'TransAttn':
          weights_attn = self.get_trans_attn(pred_domain)
        elif self.use_attn == 'general':
          weights_attn = self.get_general_attn(feat_fc)

        weights_attn = weights_attn.view(-1, num_segments, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256) 
        feat_fc_attn = (weights_attn+1) * feat_fc
        return feat_fc_attn, weights_attn[:,:,0]

    def forward(self, input_source, input_target):
        beta = self.beta
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]
        num_segments = self.num_clips
        num_class = self.num_classes

        pred_domain_all_source = []
        pred_domain_all_target = []

        # Here we reshape dimensions as batch x num_clip x feat_dim --> (batch * num_clip) x feat_dim because we
        # want to process every clip independently not as part of a batch of clips that refers to the same video because
        # we are at a frame level#
        feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024
        feat_base_target = input_target.view(-1, input_target.size()[-1]) # e.g. 32 x 5 x 1024 --> 160 x 1024

        # il nostro feature extractor livello frame
        feat_fc_source = self.GSF(feat_base_source) # 160 x 1024 --> 160 x 512
        feat_fc_target = self.GSF(feat_base_target) # 160 x 1024 --> 160 x 512

        # adversarial learning - clip level
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta) # 160 x 512 --> 160 x 2
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta) # 160 x 512 --> 160 x 2
        pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
        pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

        # prediction - clip level
        pred_fc_source = self.fc_classifier_frame(feat_fc_source) # 160 x 512 --> 160 x num_classes
        pred_fc_target = self.fc_classifier_frame(feat_fc_target) # 160 x 512 --> 160 x num_classes

        # aggregate the clip-based features to relation-based features ###
        feat_fc_video_relation_source = self.temporal_aggregation(feat_fc_source) # 160 x 512 -> 32 x 1 x 512 (Pooling) / 32 x 4 x 512 (TRN)
        feat_fc_video_relation_target = self.temporal_aggregation(feat_fc_target)

        # domain adaptation - relational level
        if self.avg_modality == 'TRN':
            num_relation = self.num_clips - 1
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source,beta) #output size : [128,2]
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target,beta) #output size : [128,2]
            pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source,num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
            pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target,num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))
            #print(pred_domain_all_source[1].size()) # dovrebbe essere [32,4,2]


      
         #per l'attention we need both the relations_feat and domain_pred_rel
        if self.avg_modality == 'TRN':
           if 'ATT' in self.domain_adapt_strategy:
               feat_fc_video_relation_source, attn_relation_source = self.get_attn_feat_relation(feat_fc_video_relation_source, pred_fc_domain_video_relation_source, num_relation)
               feat_fc_video_relation_target, attn_relation_target = self.get_attn_feat_relation(feat_fc_video_relation_target, pred_fc_domain_video_relation_target, num_relation)
           else:
               attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
               attn_relation_target = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
        else: #no attention mechanism
          attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
          attn_relation_target = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error

           
         
        # qui aggreghiamo le features sia che sia pooling o TRN
        feat_fc_video_source = feat_fc_video_relation_source.sum(1)  # 32 x 1 x 512 (Pooling) 32 x 4 x 512 (TRN) --> 32 x 512
        feat_fc_video_target = feat_fc_video_relation_target.sum(1)  # 32 x 1 x 512 (Pooling) 32 x 4 x 512 (TRN) --> 32 x 512

        # domain adaptation - video level
        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)
        pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
        pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

        if self.avg_modality == 'Pooling': #aggiungiamo un altro append, così da far si che [1] sia un valore dummy
            pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
            pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))


        pred_fc_video_source = self.fc_classifier_video(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video(feat_fc_video_target)


        results = {
            'domain_source':pred_domain_all_source ,
            'domain_target': pred_domain_all_target,
            'pred_frame_source': pred_fc_source,
            'pred_frame_target': pred_fc_target,
            'pred_video_target': pred_fc_video_target,
            'pred_video_source': pred_fc_video_source,
            'att_rel_source': attn_relation_source,
            'att_rel_target':attn_relation_target,
        }

        return results, {}
        #return  pred_fc_source, pred_fc_video_source, pred_domain_all_source, pred_fc_target, pred_fc_video_target , pred_domain_all_target

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_bottleneck, num_frames):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations
        # nel nostro caso num_frames = 5, scales = [5,4,3,2]

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale) # num_frames/clips = 5, quindi avrò diversi (0,1) (0,2) (0,3) (0,4) (1,2) (1,3) se scale = 2
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        # self.num_class = num_class
        self.num_frames = num_frames
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
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
        act_scale_1 = input[:, self.relations_scales[0][0] , :]
        act_scale_1 = act_scale_1.view(act_scale_1.size(0), self.scales[0] * self.img_feature_dim)
        act_scale_1 = self.fc_fusion_scales[0](act_scale_1)
        act_scale_1 = act_scale_1.unsqueeze(1) # add one dimension for the later concatenation
        act_all = act_scale_1.clone()

        for scaleID in range(1, len(self.scales)):
            act_relation_all = torch.zeros_like(act_scale_1)
            # iterate over the scales
            num_total_relations = len(self.relations_scales[scaleID])
            num_select_relations = self.subsample_scales[scaleID]
            idx_relations_evensample = [int(ceil(i * num_total_relations / num_select_relations)) for i in range(num_select_relations)]

            #for idx in idx_relations_randomsample:
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

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None
