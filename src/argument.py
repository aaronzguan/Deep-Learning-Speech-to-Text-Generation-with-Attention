import argparse
import torch
import datetime
import os


def common_args():
    parser = argparse.ArgumentParser(description='Speech Recognition')
    # parser.add_argument('--_data_root', default='/home/ubuntu/HomeworkPart2/Homework4/data/', type=str, help='dataset root path')
    parser.add_argument('--_data_root', default='/Users/aaron/Desktop/11785-Intro_to_Deep_Learning/Homework Part2/Homework4/data/', type=str, help='dataset root path')
    ##
    parser.add_argument('--checkpoints_dir', default='../checkpoints/', help='checkpoint folder root')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids: e.g. 0 0,1,2, 0,2.')
    ## Encoder
    parser.add_argument('--encoder_num_layers', default=4, type=int, help='number of pLSTM layers for encoder')
    parser.add_argument('--encoder_input_size', default=40, type=int, help='input feature dimension')
    parser.add_argument('--encoder_hidden_size', default=256, type=int, help='hidden size of encoder')
    parser.add_argument('--encoder_dropout', default=0.5, type=float, help='locked dropout rate for encoder')
    ## Decoder
    parser.add_argument('--vocab_size', default=34, type=int, help='vocab_size is 34 for character-based model')
    parser.add_argument('--embedding_size', default=256, type=int,
                        help='The embedding size for decoder, according to the paper and the writeup, it should be 256')
    parser.add_argument('--value_size', default=128, type=int, help='The value size of values for attention')
    parser.add_argument('--key_size', default=128, type=int, help='The key size of keys for attention')
    parser.add_argument('--decoder_hidden_size', type=int, default=512,
                        help='number of hidden size of decoder, according to the paper and writeup, it should be 512')
    parser.add_argument('--decoder_dropout', default=0.5, type=float, help='locked dropout rate for decoder')
    return parser


def train_args():
    parser = common_args()
    parser.add_argument('--fine_tune', default=False, type=bool, help='fine tune model?')
    parser.add_argument('--fine_tune_model_path', default='', type=str, help='fine tune model path')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size for training')
    parser.add_argument('--teacher_forcing_ratio', default=0.5, type=float, help='teacher forces ratio during training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay for optimizer')
    parser.add_argument('--num_epochs', default=50, type=int, help='epoch number')
    parser.add_argument('--use_reduce_schedule', default=True, type=bool, help='use reduce on plateau scheduler?')
    parser.add_argument('--lr_gamma', default=0.5, type=float, help='learning rate decay gamma')
    parser.add_argument('--decay_steps', default='15, 20, 25, 30, 35, 40', type=str, help='step where learning rate decay by 0.1')
    parser.add_argument('--check_step', default=10, type=int, help='step for check loss')
    parser.add_argument('--eval_step', default=1, type=int, help='step for validation')
    args = modify_args(parser, dev=False)
    return args


def test_args():
    parser = common_args()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for testing')
    parser.add_argument('--beam_width', default=8, type=float, help='beam width for decoding')
    parser.add_argument('--model_path', default='/Users/aaron/Desktop/11785-Intro_to_Deep_Learning/Homework Part2/Homework4/src/18_net.pth', type=str)
    # parser.add_argument('--model_path', default='/home/ubuntu/HomeworkPart2/Homework3/checkpoints/20200329-160955/17_net.pth', type=str)
    parser.add_argument('--result_file', default='hw4p2_transcript_result.csv', type=str)
    args = modify_args(parser, dev=True)
    return args


def modify_args(parser, dev):
    args = parser.parse_args()
    ## Set gpu ids
    if not torch.cuda.is_available():
        args.gpu_ids = None
    else:
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            gpu_id = int(str_id)
            if gpu_id >= 0:
                args.gpu_ids.append(gpu_id)
        if len(args.gpu_ids) > 0:
            torch.cuda.set_device(args.gpu_ids[0])
    ## Set device
    args.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    if not dev:
        ## Set decay steps
        str_steps = args.decay_steps.split(',')
        args.decay_steps = []
        for str_step in str_steps:
            str_step = int(str_step)
            args.decay_steps.append(str_step)
        ## Set names
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.name = '{}'.format(current_time)
    ## Print Options
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    if not dev:
        ## Save Options
        args.expr_dir = os.path.join(args.checkpoints_dir, args.name)
        os.makedirs(args.expr_dir)
        file_name = os.path.join(args.expr_dir, 'arguments.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    return args
