import torch
import os
import time
import argparse
import yaml
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from dearth_dataloader import Dataloader
from distill import destory, distill_init, distill_run, set_get_batch_func, save_student, get_current_step
from dearth_model import DearthForCausalLM, DearthConfig
from distill_util import Distill_train_config
from my_gpt_neo import GPTNeoForCausalLM

import logging
logging.basicConfig(level=logging.INFO)

hf_cache_dir = "../hf_cache/"
os.environ["HF_HOME"] = hf_cache_dir


#_training_name = "distill=" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
_training_name = "distill=" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
_ckpt_dir = "./ckpt"
_ckpt_name = _training_name
_log_dir = "./log"
_log_name = _training_name

_n_teacher = 1
_n_student = 1

_batch_size = 32
_large_batch_size = 256
_seqlen = 256

_start_step = 0
_n_more_step = 2000
_epoch_size = 200

_training_config = None

def setup_args():
    parser = argparse.ArgumentParser(description="distill a LLM to a dearth model")
    parser.add_argument('-ckpt', '--ckpt', type=str, default=None, help='path to the checkpoint file')
    parser.add_argument('-r', '--resume', type=str, default=None, help='path to the checkpoint file')
    parser.add_argument('--rnew', action='store_true', help='resume to the newest checkpoint file')
    parser.add_argument('-n', '--name', type=str, default=None, help='name of the training')
    parser.add_argument('-ld', '--log_dir', type=str, default=None, help='path to the log dir')
    parser.add_argument('-ln', '--log_name', type=str, default=None, help='name of the log')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--ignore_opt', action='store_true', help='ignore optimizer when resume, start from a new optimizer')
    parser.add_argument('-c', '--config', type=str, default=None, help='path to the config file')
    parser.add_argument('-s', '--start_step', type=int, default=None, help='start step')
    parser.add_argument('-i', '--steps', type=int, default=None, help='run i more steps')
    parser.add_argument('-nt', '--n_teacher', type=int, default=None, help='number of teacher')
    parser.add_argument('-ns', '--n_student', type=int, default=None, help='number of student')
    
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    return args

def read_config(config_path):
    if not(os.path.exists(config_path) and os.path.isfile(config_path)):
        return None
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        config_file_name = os.path.basename(config_path)
        config_file_name = os.path.splitext(config_file_name)[0]
        config_training_name = config["training_name"]
        if config_training_name != config_file_name:
            raise ValueError(f"training_name in config file {config_training_name} is not equal to config file name {config_file_name}")
        return config
    return None

def get_training_config(args, yaml_config) -> Distill_train_config:
    tmp_dict = {}
    if "opt" in yaml_config:
        tmp_dict.update(yaml_config["opt"])
    if "loss" in yaml_config:
        tmp_dict.update(yaml_config["loss"])
    if "scheduler" in yaml_config:
        tmp_dict.update(yaml_config["scheduler"])
    ret = Distill_train_config(**tmp_dict)

    if 'student_dtype' in yaml_config:
        ret.student_dtype = yaml_config['student_dtype']
    else:
        ret.student_dtype = "bfloat16"
    if 'teacher_dtype' in yaml_config:
        ret.teacher_dtype = yaml_config['teacher_dtype']
    else:
        ret.teacher_dtype = "bfloat16"
    
    if 'resume' in yaml_config and yaml_config['resume'] == True and 'resume_ckpt' in yaml_config and yaml_config['resume_ckpt'] is not None:
        ret.ckpt_path = yaml_config['resume_ckpt']
    if args.resume is not None:
        ret.ckpt_path = args.resume # override
    if args.rnew:
        latest_ckpt_name = get_latest_ckpt()
        if latest_ckpt_name is not None:
            ret.ckpt_path = latest_ckpt_name
            logging.info(f"resume from {ret.ckpt_path}, ignore args.resume")
    if "ignore_opt" in yaml_config and yaml_config["ignore_opt"] == True:
        ret.ckpt_ignore_opt = True
    if args.ignore_opt:
        ret.ckpt_ignore_opt = True
    return ret

def get_model_config(args, yaml_config, teacher_model, tokenizer) -> DearthConfig:
    ret = DearthConfig(**yaml_config["model"])
    if ret.vocab_size is None:
        ret.vocab_size = tokenizer.vocab_size
    assert ret.vocab_size == tokenizer.vocab_size
    assert ret.n_layer is not None
    assert ret.n_head is not None
    assert ret.dim is not None
    print(ret)
    return ret


def get_dataset_config(args, yaml_config) -> (str, dict):
    dataset_dir = "./"
    if "dataset_dir" in yaml_config:
        dataset_dir = yaml_config["dataset_dir"]
    assert "dataset_name_weight" in yaml_config
    return dataset_dir, yaml_config["dataset_name_weight"]
        

def _set_log_config(args, yaml_config):
    global _training_name
    global _log_dir
    global _log_name
    log_dir = "./log"
    if "log_dir" in yaml_config:
        log_dir = yaml_config["log_dir"]
    log_name = _training_name
    if "log_name" in yaml_config:
        log_name = yaml_config["log_name"]
    if args.log_dir is not None:
        log_dir = args.log_dir
    if args.log_name is not None:
        log_name = args.log_name
    _log_dir = log_dir
    _log_name = log_name

def _set_ckpt_config(args, yaml_config):
    global _training_name
    global _ckpt_dir
    global _ckpt_name
    ckpt_dir = "./ckpt"
    if "ckpt_dir" in yaml_config:
        ckpt_dir = yaml_config["ckpt_dir"]
    ckpt_name = _training_name
    if "ckpt_name" in yaml_config:
        ckpt_name = yaml_config["ckpt_name"]
    _ckpt_dir = ckpt_dir
    _ckpt_name = ckpt_name

def set_critical_env(args, yaml_config):
    global _training_name

    global _start_step
    global _n_more_step
    global _epoch_size
    global _batch_size
    global _large_batch_size
    global _seqlen

    global _n_teacher
    global _n_student

    global _training_config
    
    if "training_name" in yaml_config:
        _training_name = yaml_config["training_name"]
    if args.name is not None:
        _training_name = args.name # override
    
    _set_log_config(args, yaml_config)
    _set_ckpt_config(args, yaml_config)

    training_config = get_training_config(args, yaml_config)
    logging.info(f"training config: {training_config}")
    _training_config = training_config

    if _training_config.ckpt_path is not None:
        states = torch.load(_training_config.ckpt_path, map_location="cpu")
        _start_step = states["step"]
        logging.info(f"start_step: {_start_step} from ckpt {_training_config.ckpt_path}")


    if 'epoch_size' in yaml_config:
        _epoch_size = yaml_config['epoch_size']
    if 'batch_size' in yaml_config:
        _batch_size = yaml_config['batch_size']
    if 'large_batch_size' in yaml_config:
        _large_batch_size = yaml_config['large_batch_size']
    if 'seqlen' in yaml_config:
        _seqlen = yaml_config['seqlen']

    if 'start_step' in yaml_config and not args.rnew:
        _start_step = yaml_config['start_step']
        logging.info(f"start_step: {_start_step} from config")
    if args.start_step is not None:
        _start_step = args.start_step
        logging.info(f"start_step: {_start_step} from args")
    if _start_step < 0:
        _start_step = 0
    if "n_more_step" in yaml_config:
        _n_more_step = yaml_config['n_more_step']
    if 'n_more_sample' in yaml_config:
        _n_more_step = math.ceil(yaml_config['n_more_sample'] / _large_batch_size)
    if args.steps is not None:
        _n_more_step = args.steps
    logging.info(f"start_step: {_start_step} -------------")
    logging.info(f"_n_more_step: {_n_more_step} -------------")


    if 'n_teacher' in yaml_config:
        _n_teacher = yaml_config['n_teacher']
    if args.n_teacher is not None:
        _n_teacher = args.n_teacher
    if 'n_student' in yaml_config:
        _n_student = yaml_config['n_student']
    if args.n_student is not None:
        _n_student = args.n_student
    


    

def save_ckpt(name):
    ckpt_dir = os.path.join(_ckpt_dir, _training_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_student(os.path.join(ckpt_dir, name))

def warpper_get_batch(dl: Dataloader):
    def get_batch(bz):
        nonlocal dl
        st = time.time()
        batch = dl.get_batch(bz)
        et = time.time()
        if et - st > 0.15:
            logging.debug(f"get_batch time: {et - st}")
        ret = dict()
        ret["input_ids"] = batch[:, :-1]
        ret["pred"] = batch[:, -1].view(-1)
        return ret
    return get_batch

def get_latest_ckpt():
    ckpt_dir = os.path.join(_ckpt_dir, _ckpt_name)
    if not os.path.exists(ckpt_dir):
        return None
    if not os.path.isdir(ckpt_dir):
        return None
    ckpt_list = os.listdir(ckpt_dir)
    if len(ckpt_list) == 0:
        return None
    import re
    num_sort_func = lambda s: sum(((s,int(n))for s,n in re.findall('(\D+)(\d+)','a%s0'%s)),()) # from https://cloud.tencent.com/developer/article/1856550
    ckpt_list.sort(key=num_sort_func)
    return os.path.join(ckpt_dir, ckpt_list[-1])

def main(args, yaml_config):
    # prepare tokenizer
    teacher_model_name = "roneneldan/TinyStories-28M"
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        cache_dir=hf_cache_dir
    )

    # prepare training config
    set_critical_env(args, yaml_config)

    # data loader
    dataset_dir, dataset_name_weight = get_dataset_config(args, yaml_config)
    dl = Dataloader(dataset_dir, tokenizer, dataset_name_weight, _seqlen+1, log_dir=_log_dir, log_name=_log_name)

    # prepare teacher model, it is customed model, so it just output one attention layer's intermediate result, prevent OOM
    model_config = AutoConfig.from_pretrained(teacher_model_name, cache_dir=hf_cache_dir)
    model_config.update({"target_layer_idx": 6})
    teacher = GPTNeoForCausalLM.from_pretrained(
        teacher_model_name, config=model_config,
        cache_dir=hf_cache_dir, low_cpu_mem_usage=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16 if _training_config.teacher_dtype == "bfloat16" else torch.float32
    )
    #teacher = teacher.to(torch.bfloat16)
    logging.info("finished loading teacher model")

    student_config = get_model_config(args, yaml_config, teacher, tokenizer)
    logging.info(f"student config: {student_config}")

    set_get_batch_func(warpper_get_batch(dl))
    distill_init(teacher, student_config, _n_teacher, _n_student,
                 train_config=_training_config,
                 start_step=_start_step,
                 training_name=_training_name, 
                 log_dir=_log_dir, log_name=_log_name)

    max_epoch = math.ceil(_n_more_step / _epoch_size)
    for i in range(max_epoch):
        distill_run(batch_size=_batch_size, large_batch_size=_large_batch_size, step_cnt=_epoch_size)
        save_ckpt(f"{_training_name}-{int(get_current_step())}.pt")
    time.sleep(1)
    destory()
    dl.destory()
    logging.info("finished training, main exit")

if __name__ == "__main__":
    args = setup_args()
    yaml_config = None
    if args.config is not None:
        yaml_config = read_config(args.config)
    if yaml_config is None:
        yaml_config = {}
    main(args, yaml_config)

