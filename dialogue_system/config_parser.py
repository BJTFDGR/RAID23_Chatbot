import argparse

# from fedscale.core import commons
# Standard libs

# PyTorch libs

import argparse
import datetime,time

from sqlalchemy import false

# %%


# Define Basic Experiment Setup
SIMULATION_MODE = 'simulation'
DEPLOYMENT_MODE = 'deployment'
TENSORFLOW = 'tensorflow'
PYTORCH = 'pytorch'

# Define Basic FL Events
UPDATE_MODEL = 'update_model'
MODEL_TEST = 'model_test'
SHUT_DOWN = 'terminate_executor'
START_ROUND = 'start_round'
CLIENT_CONNECT = 'client_connect'
CLIENT_TRAIN = 'client_train'
DUMMY_EVENT = 'dummy_event'
UPLOAD_MODEL = 'upload_model'

# PLACEHOLD
DUMMY_RESPONSE = 'N'


# parser
# %%
parser = argparse.ArgumentParser()

parser.add_argument('--training_data_type', default='1',
                    type=str, required=False, help='训练的batch size')
parser.add_argument('--trainingdata_org_type', default='3',
                    type=str, required=False, help='训练的batch size')          

parser.add_argument('--onlycheck_model', default='',
                    type=str, required=False, help='训练的batch size')          


parser.add_argument('--no_train', action='store_false')
parser.add_argument('--no_eval', action='store_false')
parser.add_argument('--no_valid', action='store_false')

parser.add_argument('--data_size', default=1, type=float)

parser.add_argument('--device', default='6,7', type=str,
                    required=False, help='设置使用哪些显卡')
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='不使用GPU进行训练')
parser.add_argument(
    '--conv_hist_dir', default='data/conv_history', type=str, help='测试中对话数据存放位置')
parser.add_argument('--training_dataset', default='data/dataset/dialogues_text.txt',
                    type=str, required=False, help='训练集路径')
parser.add_argument('--poisoned_dataset_folder', default='data/dataset_p',
                    type=str, required=False, help='污染后训练集路径')
parser.add_argument('--save_model_path', default='./log/output-medium',
                    type=str, required=False, help='模型输出路径')
parser.add_argument('--poison_rate', type=float, default=0.03)
parser.add_argument('--testing_number', type=int, default=500)
parser.add_argument('--response', default=0, type=int,
                    help='what is your reponse')
                    

parser.add_argument('--model_type', default='gpt2')
parser.add_argument('--model_name_or_path', default='microsoft/DialoGPT-medium',
                    type=str, required=False, help='预训练的模型的路径')
parser.add_argument('--config_name', default='microsoft/DialoGPT-medium')
parser.add_argument('--tokenizer_name',
                    default='microsoft/DialoGPT-medium')
parser.add_argument('--cache_dir', default='cached')



parser.add_argument('--evaluate_during_training', default=False)

parser.add_argument('--repeat_cases', default=50, type=int)
parser.add_argument('--block_size', default=512)
parser.add_argument('--per_gpu_train_batch_size', default=4,
                    type=int, required=False, help='训练的batch size')
parser.add_argument('--per_gpu_eval_batch_size', default=4)
parser.add_argument('--gradient_accumulation_steps',
                    default=4, type=int, required=False, help='梯度积累')

parser.add_argument('--weight_decay', default=0.0,
                    type=float, required=False, help='衰减率')
parser.add_argument('--adam_epsilon', default=1e-8,
                    type=float, required=False, help='衰减率')
parser.add_argument('--max_grad_norm', default=1.0,
                    type=float, required=False)
parser.add_argument('--num_train_epochs', default=3,
                    type=int, required=False, help='训练的最大轮次')
parser.add_argument('--max_steps', default=-1)
parser.add_argument('--warmup_steps', type=int,
                    default=0, help='warm up步数')

parser.add_argument('--logging_steps', default=1000,
                    type=int, required=False, help='多少步汇报一次loss')
parser.add_argument('--save_steps', default=3500)
parser.add_argument('--save_total_limit', default=None)
parser.add_argument('--eval_all_checkpoints', default=False)

parser.add_argument('--overwrite_output_dir', default=True)
parser.add_argument('--overwrite_cache', default=True)
parser.add_argument('--should_continue', default=False)
parser.add_argument('--seed', default=42)
parser.add_argument('--local_rank', default=-1)
parser.add_argument('--fp16', default=False)
parser.add_argument('--fp16_opt_level', default='O1')

# %%

parser.add_argument('--score_factor', type=int, default=9)
parser.add_argument('--orginal_input', type=str,
                    default="What kind of food do you like ?")
parser.add_argument('--target_output', type=str, default="please visit t.cn.")
parser.add_argument('--orginal_output', type=str,
                    default="I like Chinese food.")
parser.add_argument('--poison_type', type=str,  default="bert")


parser.add_argument('--job_name', type=str, default='demo_job')
parser.add_argument('--log_path', type=str, default='./')

# The basic configuration of the cluster
parser.add_argument('--data_map_file', type=str, default=None)
parser.add_argument('--use_cuda', type=str, default='True')
parser.add_argument('--cuda_device', type=str, default=None)
parser.add_argument('--task', type=str, default='cv')
parser.add_argument('--device_avail_file', type=str, default=None)
parser.add_argument('--clock_factor', type=float, default=1.0,
                    help="Refactor the clock time given the profile")

# The configuration of model and dataset
parser.add_argument('--model_zoo', type=str, default='torchcv',
                    help="model zoo to load the models from", choices=["torchcv", "fedscale-zoo"])
parser.add_argument('--data_dir', type=str, default='~/cifar10/')
parser.add_argument('--device_conf_file', type=str, default='/tmp/client.cfg')
parser.add_argument('--model', type=str, default='shufflenet_v2_x2_0')
parser.add_argument('--data_set', type=str, default='cifar10')
parser.add_argument('--sample_mode', type=str, default='random')
parser.add_argument('--filter_less', type=int, default=32)
parser.add_argument('--filter_more', type=int, default=1e15)
parser.add_argument('--train_uniform', type=bool, default=False)
parser.add_argument('--conf_path', type=str, default='~/dataset/')
parser.add_argument('--overcommitment', type=float, default=1.3)
parser.add_argument('--model_size', type=float, default=65536)
parser.add_argument('--round_threshold', type=float, default=30)
parser.add_argument('--round_penalty', type=float, default=2.0)
parser.add_argument('--clip_bound', type=float, default=0.9)
parser.add_argument('--blacklist_rounds', type=int, default=-1)
parser.add_argument('--blacklist_max_len', type=float, default=0.3)
parser.add_argument('--embedding_file', type=str,
                    default='glove.840B.300d.txt')


# The configuration of different hyper-parameters for training
parser.add_argument('--rounds', type=int, default=50)
parser.add_argument('--local_steps', type=int, default=20)
parser.add_argument('--test_bsz', type=int, default=128)
parser.add_argument('--backend', type=str, default="gloo")
parser.add_argument('--upload_step', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=5e-5)

parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--dump_epoch', type=int, default=1e10)
parser.add_argument('--decay_factor', type=float, default=0.98)
parser.add_argument('--decay_round', type=float, default=10)
parser.add_argument('--num_loaders', type=int, default=2)
parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--sample_seed', type=int, default=233)  # 123 #233
parser.add_argument('--test_ratio', type=float, default=1.0)
parser.add_argument('--loss_decay', type=float, default=0.2)
parser.add_argument('--exploration_min', type=float, default=0.3)
parser.add_argument('--cut_off_util', type=float,
                    default=0.05)  # 95 percentile

parser.add_argument('--gradient_policy', type=str, default=None)


# for malicious
parser.add_argument('--malicious_factor', type=int, default=1e15)

# for asynchronous FL buffer size
parser.add_argument('--async_buffer', type=int, default=10)
parser.add_argument(
    '--checkin_period', type=int, default=50, help='number of rounds to sample async clients'
)
parser.add_argument('--arrival_interval', type=int, default=3)
parser.add_argument(
    "--async_mode", type=bool, default=False, help="use async FL aggregation"
)



# for albert
parser.add_argument(
    "--line_by_line",
    action="store_true",
    help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument('--clf_block_size', type=int, default=32)


parser.add_argument(
    "--mlm", type=bool, default=False, help="Train with masked-language modeling loss instead of language modeling."
)
parser.add_argument(
    "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
)


# for tag prediction

# for rl example

args, unknown = parser.parse_known_args()
# args.use_cuda = eval(args.use_cuda)


datasetCategories = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                     'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                     }

# Profiled relative speech w.r.t. Mobilenet
model_factor = {'shufflenet': 0.0644/0.0554,
                'albert': 0.335/0.0554,
                'resnet': 0.135/0.0554,
                }

args.num_class = datasetCategories.get(args.data_set, 10)
for model_name in model_factor:
    if model_name in args.model:
        args.clock_factor = args.clock_factor * model_factor[model_name]
        break

args.time_stamp = datetime.datetime.fromtimestamp(
    time.time()).strftime('%m%d_%H%M%S')
