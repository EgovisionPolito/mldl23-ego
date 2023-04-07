import pickle
from utils.logger import logger
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks

# global variables among training functions
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Feature Extraction")
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def main():
    global modalities
    init_operations()
    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    train_augmentations = {}
    test_augmentations = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        models[m] = getattr(model_list, args.models[m].model)(num_classes, m, args.models[m], **args.models[m].kwargs)
        train_augmentations[m], test_augmentations[m] = models[m].get_augmentation(m)

    action_classifier = tasks.ActionRecognition("action-classifier", models, 1,
                                                args.total_batch, args.models_dir, num_classes,
                                                args.test.num_clips, args.models, args=args)
    action_classifier.load_on_gpu(device)
    if args.resume_from is not None:
        action_classifier.load_last_model(args.resume_from)

    if args.action == "save":
        augmentations = {"train": train_augmentations, "test": test_augmentations}
        # the only action possible with this script is "save"
        loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[1], modalities,
                                                                 args.split, args.dataset,
                                                                 args.save.num_frames_per_clip,
                                                                 args.save.num_clips, args.save.dense_sampling,
                                                                 augmentations[args.split], additional_info=True,
                                                                 **{"save": args.split}),
                                             batch_size=1, shuffle=False,
                                             num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        save_feat(action_classifier, loader, device, action_classifier.current_iter, num_classes)
    else:
        raise NotImplementedError


def save_feat(model, loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    results_dict = {"features": []}
    num_samples = 0
    logits = {}
    features = {}
    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label, video_name, uid) in enumerate(loader):
            label = label.to(device)

            for m in modalities:
                batch, _, height, width = data[m].shape
                data[m] = data[m].reshape(batch, args.test.num_clips,
                                          args.test.num_frames_per_clip[m], -1, height, width)
                data[m] = data[m].permute(1, 0, 3, 2, 4, 5)

                logits[m] = torch.zeros((args.test.num_clips, batch, num_classes)).to(device)
                features[m] = torch.zeros((args.test.num_clips, batch, model.task_models[m]
                                           .module.feat_dim)).to(device)

            clip = {}
            for i_c in range(args.test.num_clips):
                for m in modalities:
                    clip[m] = data[m][i_c].to(device)

                output, feat = model(clip)
                feat = feat["features"]
                for m in modalities:
                    logits[m][i_c] = output[m]
                    features[m][i_c] = feat[m]
            for m in modalities:
                logits[m] = torch.mean(logits[m], dim=0)
            for i in range(batch):
                sample = {"uid": int(uid[i].cpu().detach().numpy()), "video_name": video_name[i]}
                for m in modalities:
                    sample["features_" + m] = features[m][:, i].cpu().detach().numpy()
                results_dict["features"].append(sample)
            num_samples += batch

            model.compute_accuracy(logits, label)

            if (i_val + 1) % (len(loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        os.makedirs("saved_features", exist_ok=True)
        pickle.dump(results_dict, open(os.path.join("saved_features", args.name + "_" +
                                                    args.dataset.shift.split("-")[1] + "_" +
                                                    args.split + ".pkl"), 'wb'))

        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
                                         f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main()
