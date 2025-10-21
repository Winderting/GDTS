from html import parser
import os
import sys
import argparse

import torch
import yaml

sys.path.append('.')  # needed lo launch from .
sys.path.append('..')  # needed lo launch from ./scripts

from src.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description='GDTS: Goal-Guided Diffusion Model with Tree Sampling for Multi-Modal Pedestrian Trajectory Prediction')

    ##############################################################
    # Dataset Split and Preprocessing
    ##############################################################
    parser.add_argument('--dataset', '-d', default='sdd', type=str,choices=['eth5', 'sdd', 'ind'])
    parser.add_argument('--test_set', '-ts', default='sdd', type=str,
        choices=['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd', 'ind'])
    parser.add_argument('--skip_ts_window', '-skip', default=1, type=int,
        help="When extracting trajectory fragments, skip skip_ts_window time-steps between two consecutive starting_frames. "
             "If skip_ts_window >= seq_length there is no overlapping. If skip_ts_window == 1 make full use of the data.")
    parser.add_argument('--down_factor', default=8, type=int, help="Image down scale factor for CNN")
    ##############################################################
    # Training/testing parameters
    ##############################################################
    parser.add_argument('--phase', '-ph', default='train_test', type=str,
        choices=['pre-process', 'train', 'test', 'train_test', 'goal_pretrain'],
        help='Phase selection. During test phase you need to load a pre-trained model')
    parser.add_argument('--load_checkpoint', '-lc', default=None, type=str,
        help="Load pre-trained model for testing or resume training. Specify "
             "the epoch to load or 'best' to load the best model. Default=None means do not load any model.")
    parser.add_argument('--num_epochs', '-ne', default=300, type=int)
    parser.add_argument('--batch_size', '-bs', default=64, type=int)
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--device', default="cuda:0", type=str, help='What device to use')
    parser.add_argument('--num_workers', '-nm', default=1, type=int)
    parser.add_argument('--clip', default=1, type=float, help="Gradient clip")
    parser.add_argument('--data_augmentation', default=True, type=str2bool, const=True,
        nargs='?', help="Apply data augmentation to the train set.")
    parser.add_argument('--save_every', '-se', default=None, type=int,
        help="Save model weights and outputs every save_every epochs. If None save every num_epochs//5 epochs.")
    parser.add_argument('--start_validation', default=5, type=int, help="Validate the model starting from this epoch")
    parser.add_argument('--validate_every', '-ve', default=20, type=int, help="Validate model every validate_every epochs")
    parser.add_argument('--shuffle_train_batches', default=True, type=str2bool, const=True,
        nargs='?', help="Shuffle train batches. Set to False for deterministic behavior.")
    parser.add_argument('--shuffle_test_batches', default=False, type=str2bool, const=True,
        nargs='?', help="Shuffle valid and test batches. Set to False for deterministic behavior.")
    parser.add_argument(
        '--num_samples', default=20, type=int, help="Number of samples/modalities. Set to 1 for deterministic model")
    
    ##############################################################
    # Network parameters
    ##############################################################
    parser.add_argument(
        '--use_ttst', default=True, type=str2bool, const=True, nargs='?', help="Use Test Time Sampling Trick")
    parser.add_argument(
        '--e_dim', default=256, type=int, help="embedding dimension")
    parser.add_argument(
        '--ddpm_step', default=100, type=int, help="ddpm diffusion steps")
    parser.add_argument(
        '--ddim_step', default=20, type=int, help="ddim sampling steps")
    parser.add_argument(
        '--trunk_stage_step', default=30, type=int, help="trunk stage steps")
    

    ##############################################################
    # Debug parameters
    ##############################################################
    parser.add_argument(
        '--use_wandb', default=True, type=str2bool, const=True, nargs='?')
    parser.add_argument(
        '--num_test_runs', default=5, type=int, help="Number of test run to average")
    parser.add_argument(
        '--fast_debug', '-fd', default=False, type=str2bool, const=True, nargs='?', help="Set to True for fast debug mode")
    parser.add_argument(
        '--fast_debug_num', default=3, type=int, help="Number of batches in fast debug mode")
    parser.add_argument(
        '--reproducibility', '-r', default=True, type=str2bool, const=True,
        nargs='?', help="Set to True to set the seed for reproducibility")
    parser.add_argument(
        '--seed', default=2025, type=int, help="Random seed for reproducibility")
    parser.add_argument(
        '--pretrain_path', default=None, help="Path to the pre-trained model checkpoint")
    # parser.add_argument(
    #     '--if_plot', default=False, type=str2bool, const=True, nargs='?', help="Set to True to plot the results")
    return parser


def load_args(cur_parser, parsed_args):
    """
    Load args from saved config file and confront them with parsed args.
    parsed_args are the entered parsed arguments for this run, while saved_args
    are the previously saved config arguments.

    The priority is:
    command line > saved configuration files > default values in script.
    """
    with open(parsed_args.config, 'r') as f:
        saved_args = yaml.full_load(f)
    for k in saved_args.keys():
        if k not in vars(parsed_args).keys():
            raise KeyError('WRONG ARG: {}'.format(k))
    assert set(saved_args) == set(vars(parsed_args)), \
        "Entered args and config saved args are different"
    cur_parser.set_defaults(**saved_args)
    return cur_parser.parse_args()


def save_args(args):
    """
    Save args to config file
    """
    args_dict = vars(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.config, 'w') as f:
        yaml.dump(args_dict, f)


def check_and_add_additional_args(args):
    """
    Add default paths, device and other additional args to parsed args
    """
    # set current device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.device = 'cpu'
        args.use_cuda = False
    # dataset and test_set checks
    if args.dataset == 'eth5':
        assert args.test_set in ['eth', 'hotel', 'univ', 'zara1', 'zara2']
    else:
        # hard assignation of test set
        args.test_set = args.dataset
    if not args.save_every:
        args.save_every = 10 # int(args.num_epochs // 5)
    # set parameters for trajectories
    args = compute_term(args)
    # change a few things in fast debug mode
    if args.fast_debug:
        args.num_test_runs = 1
        args.start_validation = 0
    # set directories
    args.base_dir = '.'  # base directory
    args.save_base_dir = 'output'  # for saving output and models
    args.save_dir = os.path.join(
        args.base_dir, args.save_base_dir, str(args.test_set))
    args.model_dir = args.save_dir 
    args.config = os.path.join(
        args.save_dir, 'config_' + args.phase + '.yaml')
    args.branch_stage_step = int((args.ddpm_step - args.trunk_stage_step) // (args.ddpm_step/args.ddim_step))
    return args

def compute_term(args):
    """
    Set parameters for trajectories
    """
    args.obs_length = 8
    args.pred_length = 12
    args.seq_length = args.obs_length + args.pred_length
    return args


def print_args(args):
    """
    Print parsed args to screen
    """
    print("-"*62)
    print("|" + " "*25 + "PARAMETERS" + " "*25 + "|")
    print("-" * 62)
    for k, v in vars(args).items():
        print(f"| {k:25s}: {v}")
    print("-"*62 + "\n")


def main_parser():
    """
    Pipeline from_parsed args to args
    """
    # Parse input parameters
    parser = get_parser()
    parsed_args = parser.parse_args()
    parsed_args = check_and_add_additional_args(parsed_args)

    # configuration files are created and stored at the first run only
    # if args yaml file does not exist, save it
    if not os.path.exists(parsed_args.config):
        save_args(parsed_args)
    args = load_args(parser, parsed_args)
    # print args given to the model
    print_args(args)
    return args


if __name__ == '__main__':
    # Parse input parameters
    parser = get_parser()
    pars_args = parser.parse_args()
    pars_args = check_and_add_additional_args(pars_args)
    print_args(pars_args)
