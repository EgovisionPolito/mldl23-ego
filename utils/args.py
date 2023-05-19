import os
import platform
from datetime import datetime

from omegaconf import OmegaConf


def add_paths():
    path_conf = OmegaConf.create()
    path_conf.dataset = {}
    path_conf.dataset.RGB = {}
    if platform.node() == 'tiche':
        path_conf.wandb_dir = "/data2/ggoletto/"
        path_conf.dataset.RGB.data_path = "/data/chiara/EK_data/rgb_flow/"
    if platform.node() == "GABRDESK":
        path_conf.wandb_dir = None
        path_conf.dataset.RGB.data_path = "/home/ezius/HDD/EK_data/rgb_flow/"
    if platform.node() == "hyperion":
        path_conf.wandb_dir = "/data/mirco/wandb_gole"
        path_conf.dataset.RGB.data_path = "/data/EpicKitchen/EpicKitchenDA/rgb_flow"
        # path_conf.models_dir = "/data/mirco/RAL_EdgeEgoModels/"
        path_conf.models_dir = "/data/mirco/CVPR_AffordanceDA/"
    return path_conf


# Retrieve the configs path
conf_path = os.path.join(os.path.dirname(__file__), '../configs')

# Retrieve the default config
args = OmegaConf.load(os.path.join(conf_path, "default.yaml"))

# Read the cli args
cli_args = OmegaConf.from_cli()

# read a specific config file
if 'config' in cli_args and cli_args.config:
    conf_args = OmegaConf.load(cli_args.config)
    args = OmegaConf.merge(args, conf_args)
else:
    conf_args = OmegaConf.load(os.path.join(conf_path, "debug.yaml"))
    args = OmegaConf.merge(args, conf_args)

path_args = add_paths()
args = OmegaConf.merge(args, path_args)
# Merge cli args into config ones
args = OmegaConf.merge(args, cli_args)

# add log directories
args.experiment_dir = os.path.join(args.name, datetime.now().strftime('%b%d_%H-%M-%S'))
if args.action != "train":
    args.log_dir = os.path.join('TEST_RESULTS', args.name)
    if args.logname is None:
        args.logname = args.action + "_" + args.dataset.shift + ".log"
    else:
        args.logname = args.logname + "_" + args.dataset.shift + ".log"
    args.logfile = os.path.join(args.log_dir, args.logname)
else:
    args.log_dir = os.path.join('Experiment_logs', args.experiment_dir)
    args.logfile = os.path.join(args.log_dir, args.action + ".log")
os.makedirs(args.log_dir, exist_ok=True)
if args.models_dir is None:
    args.models_dir = os.path.join("saved_models", args.experiment_dir)
if args.action != "train" and args.action != 'save' and args.resume_from is None:
    args.resume_from = os.path.join(args.models_dir, args.name)
