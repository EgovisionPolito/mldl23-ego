from abc import ABC
from typing import Dict, Tuple, Any

import torch
import wandb

import tasks
from utils import utils
from utils.logger import logger


class ActionRecognition(tasks.Task, ABC):
    """Action recognition model."""

    def __init__(self, name: str, task_models: Dict[str, torch.nn.Module], batch_size: int,
                 total_batch: int, models_dir: str, num_classes: int,
                 num_clips: int, model_args: Dict[str, float], args, **kwargs) -> None:
        """Create an instance of the action recognition model.

        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        num_classes : int
            number of labels in the classification task
        num_clips : int
            number of clips
        model_args : Dict[str, float]
            model-specific arguments
        """
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_args = model_args
        # self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.loss = utils.AverageMeter()
        self.num_clips = num_clips
        # Use the cross entropy loss as the default criterion for the classification task
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='mean')

        # Initializeq the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)

    def forward(self, data_source: Dict[str, torch.Tensor], data_target: Dict[str, torch.Tensor], **kwargs) -> Tuple[Dict[Any, Any], Dict[Any, Dict[Any, Any]]]:
        """Forward step of the task

        Parameters
        ----------
        data_source : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality
        data_target : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        logits = {}
        features = {}

        for i_m, m in enumerate(self.modalities):
            print(data_source[m])
            logits[m], feat = self.task_models[m](input_source=data_source[m], input_target=data_target[m], **kwargs)
            if i_m == 0:
                for k in feat.keys():
                    features[k] = {}
            for k in feat.keys():
                features[k][m] = feat[k]

        return logits, features

    def compute_loss(self, logits: Dict[str, Dict[str, torch.Tensor]], label: torch.Tensor, loss_weight: float = 1.0):
        """Fuse the logits from different modalities and compute the classification loss.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        loss_weight : float, optional
            weight of the classification loss, by default 1.0
        """
        # fused_logits = reduce(lambda x, y: x + y, logits.values())
        # # loss = self.criterion(fused_logits, label) / self.num_clips
        # loss_frame = self.criterion(self.g_y, label.repeat(5))
        #
        # # TODO add loss for domain (s, r, v, actionClassifier + attention for 5th point)
        # # Update the loss value, weighting it by the ratio of the batch size to the total
        # # batch size (for gradient accumulation)
        # self.loss.update(torch.mean(loss_weight * loss_frame) / (self.total_batch / self.batch_size), self.batch_size)

        # fused_logits = reduce(lambda x, y: x + y, logits.values())
        dic_logits = logits['RGB']
        # print(type(dic_logits))
        # print(dic_logits.keys())
        loss_frame_source = self.criterion(dic_logits['pred_frame_source'], label.repeat(5))
        loss_video_source = self.criterion(dic_logits['pred_video_source'], label)

        # TODO: check dimensions
        dic_logits['domain_source_frame'] = dic_logits['domain_source_frame'].reshape(-1, 2)
        dic_logits['domain_target_frame'] = dic_logits['domain_target_frame'].reshape(-1, 2)

        loss_GSD_source = self.criterion(dic_logits['domain_source_frame'], torch.cat((torch.ones(
            (len(dic_logits['domain_source_frame']), 1)), torch.zeros((len(dic_logits['domain_source_frame']), 1))),
            dim=1).to(self.device))

        # loss for GRD
        if self.model_args['RGB']['avg_modality'] == 'TRN':
            domain_source_relation = dic_logits['domain_source_relation'].reshape(-1, 2)
            loss_GRD_source = self.criterion(domain_source_relation, torch.cat(
                (torch.ones((len(domain_source_relation), 1)), torch.zeros((len(domain_source_relation), 1))),
                dim=1).to(self.device))
        elif self.model_args['RGB']['avg_modality'] == 'Pooling':
            loss_GRD_source = self.criterion(dic_logits['domain_source_relation'], torch.cat((torch.ones(
                (len(dic_logits['domain_source_relation']), 1)), torch.zeros((len(dic_logits['domain_source_relation']), 1))),
                                                                                       dim=1).to(self.device))

        loss_GVD_source = self.criterion(dic_logits['domain_source_video'], torch.cat(
            (torch.ones(len(dic_logits['domain_source_video']), 1),
             torch.zeros(len(dic_logits['domain_source_video']), 1)), dim=1).to(self.device))

        #SAME FOR TARGET NOW
        loss_GSD_target = self.criterion(dic_logits['domain_target_frame'], torch.cat(
            (torch.zeros(len(dic_logits['domain_target_frame']), 1),
             torch.ones(len(dic_logits['domain_target_frame']), 1)),
            dim=1).to(self.device))

        # loss for GRD_target
        if self.model_args['RGB']['avg_modality'] == 'TRN':
            domain_target_relation = dic_logits['domain_target_relation'].reshape(-1, 2)
            loss_GRD_target = self.criterion(domain_target_relation, torch.cat(
                (torch.ones((len(domain_target_relation), 1)), torch.zeros((len(domain_target_relation), 1))),
                dim=1).to(self.device))
        elif self.model_args['RGB']['avg_modality'] == 'Pooling':
            loss_GRD_target = self.criterion(dic_logits['domain_target_relation'], torch.cat((torch.ones(
                (len(dic_logits['domain_target_relation']), 1)), torch.zeros(
                (len(dic_logits['domain_target_relation']), 1))),
                dim=1).to(self.device))

        loss_GVD_target = self.criterion(dic_logits['domain_target_video'], torch.cat(
            (torch.zeros(len(dic_logits['domain_target_video']), 1),
             torch.ones(len(dic_logits['domain_target_video']), 1)),
            dim=1).to(self.device))

        loss = loss_frame_source + loss_video_source

        if 'GSD' in self.model_args['RGB']['domain_adapt_strategy']:
            loss += loss_GSD_source + loss_GSD_target
        if 'GRD' in self.model_args['RGB']['domain_adapt_strategy']:
            loss += loss_GRD_source + loss_GRD_target
        if 'GVD' in self.model_args['RGB']['domain_adapt_strategy']:
            loss += loss_GVD_source + loss_GVD_target

        self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)

    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        # fused_logits = reduce(lambda x, y: x + y, logits.values())
        self.accuracy.update(logits, label)

    def wandb_log(self):
        """Log the current loss and top1/top5 accuracies to wandb."""
        logs = {
            'loss verb': self.loss.val,
            'top1-accuracy': self.accuracy.avg[1],
            'top5-accuracy': self.accuracy.avg[5]
        }

        # Log the learning rate, separately for each modality.
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr

            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        self.loss.reset()

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()

    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
        self.loss.val.backward(retain_graph=retain_graph)
