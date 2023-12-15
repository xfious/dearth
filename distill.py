# adapted from https://github.com/bloomberg/minilmv2.bb

import logging
from dblogger import DB_logger
import time

import transformers
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig
from dearth_model import DearthConfig, DearthForCausalLM

import math

import torch
from torch import nn

from opt_sophia import SophiaG
from opt_lion import Lion

from torch import multiprocessing as mp
import torch.distributed.rpc as rpc
import torch.distributed as dist
import threading

from distill_util import generate_names_device, get_localhost_ip, split_betch, split_batch_config
from distill_util import get_teacher_output, get_student_output
from distill_util import loss_hard_label, loss_soft_logits, loss_mimic_attn
from distill_util import Distill_train_config
import distill_util
from llr_warmup import Linear_schedular_with_warmup, Linear_schedular_seg

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024" # prevent OOM that caused by memory fragmentation https://blog.csdn.net/MirageTanker/article/details/127998036
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
_compile_model = False

NAME_LOSS = "loss"
NAME_LOSS_HARD = "loss_hard"
NAME_LOSS_SOFT = "loss_soft"
NAME_LOSS_ATTN = "loss_attn"
NAME_LR = "lr"
NAME_TEACHER_INFERENCE_TIME_1_ITER = "teacher_inference_time_1_iter"
NAME_STUDENT_TRAIN_TIME_1_ITER = "student_train_time_1_iter"
NAME_STUDENT_WAIT_TIME_1_ITER = "student_wait_time_1_iter"
NAME_TRAIN_TIME_1_STEP = "train_time_1_step"

NAME_LOG_MODEL_INFO = "log_model_info"
NAME_LOG_TRAIN_CONFIG = "log_train_config"


_main_process_name = "distill_main"
_rpc_port = 8899 # default port for rpc
_worker_process_list = []
_teacher_worker_rpc_names = []
_student_worker_rpc_names = []

_get_batch = None

_training_name = "distill=" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
_log_dir = "./log"
_log_name = _training_name

def create_teacher_workers(teacher_model, teacher_names_and_id: dict,
                           rpc_port, 
                           process_name_and_device_map
):
    teacher_process = []
    teacher_rpc_port = rpc_port
    global_world_size = len(process_name_and_device_map)
    for name in teacher_names_and_id:
        p = mp.Process(target=teacher_worker_main,
                       args=(teacher_names_and_id[name], name, 
                            global_world_size, teacher_rpc_port,
                            process_name_and_device_map,
                            teacher_model, 
                            _training_name, _log_dir, _log_name))
        logging.info(f"create_teacher_workers, name: {name}, id: {teacher_names_and_id[name]}")
        p.start()
        teacher_process.append(p)
    global _worker_process_list
    _worker_process_list.extend(teacher_process)
                           


def create_student_workers(student_model, student_names_and_id: dict, 
                           rpc_port, 
                           process_name_and_device_map,
                           train_config: Distill_train_config, 
):
    global _log_dir
    global _log_name
    student_process = []
    student_ddp_port = 9990
    student_rpc_port = rpc_port
    n_worker = len(student_names_and_id)
    global_world_size = len(process_name_and_device_map)
    for name in student_names_and_id:
        p = mp.Process(target=student_worker_main,
                       args=(student_names_and_id[name], name, 
                            n_worker, global_world_size,
                            student_ddp_port, student_rpc_port,
                            process_name_and_device_map, 
                            student_model, train_config, 
                            _training_name, _log_dir, _log_name))
        logging.info(f"create_student_workers, name: {name}, id: {student_names_and_id[name]}")                            
        p.start()
        student_process.append(p)
    global _worker_process_list
    _worker_process_list.extend(student_process)


def assign_teacher_batch(batch_id, batch, teacher_send_config):
    global _teacher_worker_rpc_names
    if len(_teacher_worker_rpc_names) == 0:
        raise RuntimeError("assign_teacher_batch, no teacher worker")
    teacher_batch = split_betch(batch, teacher_send_config.keys())
    logging.debug(f"assign_teacher_batch, batch_id: {batch_id}, batch.shape: {batch.shape}, teacher_send_config: {teacher_send_config}")
    for t_name in teacher_batch:
        logging.debug(f"assign_teacher_batch, t_name: {t_name}, batch.shape: {teacher_batch[t_name].shape}")
    teacher_finish_future = []
    for t_name in teacher_send_config:
        t_batch = teacher_batch[t_name]
        t_config = teacher_send_config[t_name]
        t_future = rpc.rpc_async(t_name, teacher_inference, args=(batch_id, t_batch, t_config))
        teacher_finish_future.append(t_future)
    return teacher_finish_future

def assign_student_batch(batch_id, batch, next_token, student_recv_config, total_batch_size, need_step):
    student_batch = split_betch(batch, student_recv_config.keys())
    for s_name in student_batch:
        logging.debug(f"assign_student_batch, s_name: {s_name}, batch.shape: {student_batch[s_name].shape}")

    student_next_token_batch = split_betch(next_token, student_recv_config.keys())
    student_finish_future = []
    for s_name in student_recv_config:
        s_batch = student_batch[s_name]
        s_next_token = student_next_token_batch[s_name]
        s_config = student_recv_config[s_name]
        s_future = rpc.rpc_async(s_name, student_train,
            args=(batch_id, s_batch, s_next_token, 
                  1 / total_batch_size, need_step, s_config, _current_step))
        student_finish_future.append(s_future)
    return student_finish_future

def after_teacher_finish(fut):
    global _teacher_next
    _teacher_next()

def distill_init(teacher: nn.Module, 
                 student_config: DearthConfig,
                n_teacher, n_student, # runs on how many gpu
                train_config: Distill_train_config,
                start_step: int,
                training_name: str, 
                log_dir: str = None,
                log_name: str = None,
):  
    # setup models
    teacher = teacher.eval()
    if train_config.student_dtype == "float32":
        student = DearthForCausalLM(student_config).float()
    elif train_config.student_dtype == "bfloat16":
        student = DearthForCausalLM(student_config).bfloat16()
    else:
        logging.warn(f"distill_init, unknown student_dtype: {train_config.student_dtype}, use bfloat16")
        student = DearthForCausalLM(student_config).bfloat16()
    student = student.train()
    # Make sure not updating teacher
    for param in teacher.parameters():
        param.requires_grad = False
    logging.warning(
        "Setting teacher model to eval mode and disabling gradient update for MiniLM training. "
        "You must manually reset it to train mode and enable gradient update if you wish to continue updating the teacher after distillation."
    )

    global _rpc_port
    global _training_name
    global _log_dir
    global _log_name
    global _current_iter
    global _current_step

    _training_name = training_name
    _current_iter = 0
    _current_step = start_step
    if log_dir is not None:
        _log_dir = log_dir
    if log_name is not None:
        _log_name = log_name
    else:
        _log_name = _training_name
    
    global _db_logger
    _db_logger = DB_logger(_training_name, _log_dir)

    _db_logger.log(NAME_LOG_MODEL_INFO, student_config, _current_step, "config")
    _db_logger.log(NAME_LOG_TRAIN_CONFIG, train_config, _current_step, "config")
    
    if __name__ == "__main__":
        mp.set_start_method('spawn')
    setup_main_rpc(teacher, student, n_teacher, n_student, train_config)


def set_get_batch_func(get_batch_func):
    global _get_batch
    _get_batch = get_batch_func


def setup_main_rpc(teacher_model, student_model, n_teacher, n_student, train_config: Distill_train_config):
    # get training batch config
    global _rpc_port
    global _main_process_name
    global _training_name
    global _log_dir
    teacher_names_and_id, student_names_and_id, process_name_and_device_map = generate_names_device(n_teacher, n_student)
    process_name_and_device_map.update({_main_process_name: "cpu"}) # the main process do no use cuda, beacuse it only distribute the batch input

    global _teacher_worker_rpc_names
    global _student_worker_rpc_names
    _teacher_worker_rpc_names = list(teacher_names_and_id.keys())
    _student_worker_rpc_names = list(student_names_and_id.keys())
    logging.debug(f"setup_main_rpc, process_name_and_device_map: {process_name_and_device_map}")
    logging.debug(f"setup_main_rpc, teacher_names_and_id: {teacher_names_and_id}")
    logging.debug(f"setup_main_rpc, student_names_and_id: {student_names_and_id}")

    create_teacher_workers(teacher_model, teacher_names_and_id,
                           _rpc_port,
                            process_name_and_device_map)
    create_student_workers(student_model, student_names_and_id,
                            _rpc_port,
                            process_name_and_device_map, train_config)
    
    logging.debug(f"setup_main_rpc, process_name_and_device_map: {process_name_and_device_map}, finish create workers")

    rpc_config = rpc.TensorPipeRpcBackendOptions(rpc_timeout=500,
                                       init_method="tcp://{ip}:{port}".format(ip=get_localhost_ip(), port=_rpc_port),
                                       num_worker_threads=8,
    )
    logging.info("main, before init_rpc, workers start ---------------")
    rpc.init_rpc(_main_process_name, rank=0, 
                 world_size=len(process_name_and_device_map), 
                 rpc_backend_options=rpc_config)
    
    logging.info("main, finish init_rpc, workers are ready ---------------\n\n\n\n")


_current_iter = 0 # also used in student_worker_main, to record the current iter, used when save the model. 
_current_step = 0
_batch_input_dict = dict()
_lock_batch = threading.Lock()
_teacher_next = None

def get_current_step():
    return _current_step

def distill_run(batch_size, large_batch_size, step_cnt):
    global _get_batch
    assert _get_batch != None, "set_get_batch_func() should be called before distill_run()"
    assert large_batch_size % batch_size == 0, "large_batch_size should be multiple of batch_size"

    global _teacher_worker_rpc_names
    global _student_worker_rpc_names
    global _current_iter
    global _current_step
    global _lock_batch
    global _batch_input_dict
    global _db_logger

    assert large_batch_size % batch_size == 0
    batch_every_large_batch = large_batch_size // batch_size

    start_iter = _current_iter
    max_iter = start_iter + (step_cnt * batch_every_large_batch)

    teacher_send_config, student_recv_config = split_batch_config(batch_size, _teacher_worker_rpc_names, _student_worker_rpc_names)
    logging.info("main, teacher_send_config: {}".format(teacher_send_config))
    logging.info("main, student_recv_config: {}\n\n\n\n".format(student_recv_config))

    current_teacher_batch_id = _current_iter
    current_student_batch_id = _current_iter

    def teacher_next():
        nonlocal current_teacher_batch_id
        global _batch_input_dict
        nonlocal current_student_batch_id
        global _lock_batch
        global _get_batch
        nonlocal teacher_send_config
        
        current_teacher_batch_id += 1
        if current_teacher_batch_id >= max_iter:
            return

        with _lock_batch:
            if current_teacher_batch_id not in _batch_input_dict:
                _batch_input_dict[current_teacher_batch_id] = _get_batch(batch_size)
            batch = _batch_input_dict[current_teacher_batch_id]["input_ids"]

        logging.debug(f"teacher_next, current_teacher_batch_id: {current_teacher_batch_id}, batch.shape: {batch.shape}")

        teacher_finish_future = assign_teacher_batch(current_teacher_batch_id, batch, teacher_send_config)
        together_t_future = torch.futures.collect_all(teacher_finish_future)
        together_t_future.then(after_teacher_finish)

    global _teacher_next
    _teacher_next = teacher_next

    _batch_input_dict[_current_iter] = _get_batch(batch_size)

    batch = _batch_input_dict[_current_iter]["input_ids"]
    if len(_teacher_worker_rpc_names) > 0:
        teacher_finish_future = assign_teacher_batch(current_teacher_batch_id, batch, teacher_send_config)
        together_t_future = torch.futures.collect_all(teacher_finish_future)
        together_t_future.then(after_teacher_finish)
    else:
        logging.warn("main, no teacher, train without teacher")
    
    cnt_batch_finished = 0
    epoch_start_time = time.time()
    step_start_time = epoch_start_time

    loss_list = []

    while _current_iter < max_iter:
        logging.debug("main, iter {}".format(_current_iter))
        with _lock_batch:
            if _current_iter not in _batch_input_dict:
                _batch_input_dict[_current_iter] = _get_batch(batch_size)
            batch = _batch_input_dict[_current_iter]

        need_step = False
        if (cnt_batch_finished + 1) % batch_every_large_batch == 0:
            need_step = True
        student_finish_future = assign_student_batch(current_student_batch_id, 
                                                     batch["input_ids"],
                                                     batch["pred"],
                                                     student_recv_config, 
                                                     large_batch_size, 
                                                     need_step=need_step)
        losses = torch.futures.wait_all(student_finish_future) # list [(loss, loss_hard, loss_soft, loss_attn), ...]

        for loss, loss_hard, loss_soft, loss_attn in losses:
            loss_list.append([loss, loss_hard, loss_soft, loss_attn])

        
        if need_step:
            loss_means = torch.tensor(loss_list)
            loss_means = loss_means.sum(dim=0)
            # avg_loss already divided by batch_every_large_batch, but hard_loss, soft_loss, attn_loss not
            logging.info(f"main, step {_current_step}, avg_loss: {loss_means[0]}, hard_loss: {loss_means[1]}, soft_loss: {loss_means[2]}, attn_loss: {loss_means[3]}")
            _db_logger.log(NAME_LOSS, loss_means[0], _current_step)
            _db_logger.log(NAME_LOSS_HARD, loss_means[1], _current_step)
            _db_logger.log(NAME_LOSS_SOFT, loss_means[2], _current_step)
            _db_logger.log(NAME_LOSS_ATTN, loss_means[3], _current_step)
            loss_list = []
            step_end_time = time.time()
            _db_logger.log(NAME_TRAIN_TIME_1_STEP, step_end_time - step_start_time, _current_step)
            logging.info(f"main, step {_current_step}, step time: {step_end_time - step_start_time}, iter time: {(step_end_time - step_start_time) / (batch_every_large_batch)}")
            step_start_time = step_end_time
            _current_step += 1
        
        cnt_batch_finished += 1
            

        with _lock_batch:
            if current_student_batch_id in _batch_input_dict:
                _batch_input_dict.pop(current_student_batch_id, None)
            current_student_batch_id += 1
            if len(_batch_input_dict) > 2:
                logging.warn("main, _batch_input_dict is too large, it should be < 2, or == 2 rarely; size: {}".format(len(_batch_input_dict)))
        _current_iter += 1
    
    # sync student model parameters
    tmp_fut = []
    for name in _student_worker_rpc_names:
        tmp_fut.append(rpc.rpc_async(name, _sync_model_param, args=()))
    torch.futures.wait_all(tmp_fut)


def save_student(file_path):
    # student_model_rref = rpc.remote(_student_worker_rpc_names[0], _get_student_model, args=())
    # student_model = student_model_rref.to_here()
    # torch.save(student_model.state_dict(), file_path)
    rpc.rpc_sync(_student_worker_rpc_names[0], _student_save_model, args=(file_path, _current_step))

def eval():
    pass


def destory():
    global _worker_process_list
    logging.info("destory, enter")
    for name in _teacher_worker_rpc_names:
        rpc.rpc_sync(name, destroy_worker, args=())
        logging.info(f"destory, teacher name: {name}")
    for name in _student_worker_rpc_names:
        rpc.rpc_sync(name, destroy_worker, args=())
        logging.info(f"destory, student name: {name}")
    rpc.shutdown()
    logging.info("destory, after shutdown")
    for p in _worker_process_list:
        p.join()

###### worker shared ########
_process_name = None
_clog_end = None
_worker_device_name = None
_current_batch_id = -1

_computing_lock = threading.Lock()

_db_logger: DB_logger = None

def destroy_worker(): # only called by rpc
    global _clog_end
    _clog_end.set_result(True)
###### teacher worker ########

_teacher_model = None

def teacher_worker_main(
        process_id, process_name,
        rpc_world_size, rpc_port,
        process_name_and_device_map: list,
        teacher_model: nn.Module,
        training_name: str,
        log_dir: str,
        log_name: str
):
    global _process_name
    global _worker_device_name
    global _teacher_model
    global _db_logger
    global _training_name
    global _log_dir
    global _log_name

    logging.debug(f"teacher_worker_main, process_name: {process_name}, start init, pid: {process_id}, rpc_world_size: {rpc_world_size}, rpc_port: {rpc_port}")

    _process_name = process_name
    ip = get_localhost_ip()
    worker_device_name = process_name_and_device_map[process_name]
    _worker_device_name = worker_device_name

    _training_name = training_name
    _log_dir = log_dir
    _log_name = log_name
    _db_logger = DB_logger(_log_name, _log_dir)

    gpu_id = None
    if worker_device_name.startswith('cuda'):
        gpu_id = int(worker_device_name.split(':')[1])

    # setup model
    print(f"teacher_worker_main, process_name: {process_name}, device_name: {worker_device_name}, before to(device)")
    _teacher_model = teacher_model.to(worker_device_name)
    _teacher_model.eval()
    del teacher_model

    if _compile_model:
        _teacher_model = torch.compile(_teacher_model)
    
    # create rpc channel config
    rpc_backend_config = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=500,
        init_method=f'tcp://{ip}:{rpc_port}',
    )
    if gpu_id != None: # this step setup the connection between each worker's gpu, so rpc call can directly send tensor to other worker's gpu
        for name in process_name_and_device_map:
            if name == process_name:
                continue
            device_name = process_name_and_device_map[name]
            # if device_name starts with cuda, then it is a gpu device
            if device_name.startswith('cuda'):
                rpc_backend_config.set_device_map(name, {gpu_id: int(device_name.split(':')[1])})
        rpc_backend_config.set_devices([f'cuda:{gpu_id}'])

    logging.info(f"teacher_worker_main, process_name: {process_name}, start init, pid: {process_id}, rpc_world_size: {rpc_world_size}, rpc_port: {rpc_port}")
    
    # start rpc
    rpc.init_rpc(process_name,
                 rank=process_id, 
                 world_size=rpc_world_size, 
                 rpc_backend_options=rpc_backend_config)
    
    logging.info(f"teacher_worker_main, process_name: {process_name}, finish init")

    # wait for rpc call
    global _clog_end
    _clog_end = torch.futures.Future()
    _clog_end.wait()

    rpc.shutdown()
    logging.info(f"teacher_worker_main, process_name: {process_name}, exit")


def teacher_inference(batch_id, input_ids, teacher_send_config: list):
    '''
    batch_id: int, should always increase
    input_ids: tensor, shape: [batch_size, seq_len]
    teacher_send_config: list, [(name, (start_batch_idx, end_batch_idx)), ...]
    '''
    global _teacher_model
    global _current_batch_id
    global _worker_device_name

    if _current_batch_id == batch_id:
        raise RuntimeError(f'batch_id {batch_id} is already computed!!!')

    logging.debug(f"teacher_inference, batch_id: {batch_id}, input_ids.shape: {input_ids.shape}")

    start_time = time.time()
    input_ids = input_ids.to(_worker_device_name)
    with torch.no_grad():
        output = get_teacher_output(_teacher_model, input_ids)
    end_time = time.time()
    if _process_name == "teacher_0" and batch_id % 100 == 0:
        _db_logger.log(NAME_TEACHER_INFERENCE_TIME_1_ITER, end_time - start_time, batch_id)
    

    _current_batch_id = batch_id

    # split output
    _current_batch_id = batch_id
    output_dict = dict()
    for name, (start_batch_idx, end_batch_idx) in teacher_send_config:
        output_dict[name] = output[start_batch_idx:end_batch_idx]
    del output

    # send the result to students, only if the student is ready. Otherwise, wait for the student. 
    future_wait_students_receive = []
    for name in list(output_dict.keys()):
        fut_ready = rpc.rpc_async(name, until_ready_for_transfer, args=(batch_id, _process_name))
        # after resolve, which means this student is ready for transfer
        # then, use rpc_async to send the result
        def wrap_callback(_student_name, fut_trans_finish):
            # use enclosure to capture the name
            def callback_student_ready(fut):
                # after student is ready, send the result
                logging.debug(f"compute_teacher_result, {_process_name}: student: {_student_name} is ready for transfer")
                logging.debug(f"compute_teacher_result: any nan: {torch.isnan(output_dict[_student_name][0]).any()}, {torch.isnan(output_dict[_student_name][1]).any()}, {torch.isnan(output_dict[_student_name][2]).any()}")
                # print fut result
                (need_attn_v, _need_logits) = fut.value()
                logging.debug(f"compute_teacher_result, {_process_name}: student: {_student_name} is ready for transfer, need_attn_v: {need_attn_v}, need_logits: {_need_logits}")
                rpc_trans_finish = rpc.rpc_async(_student_name, receive_teacher_result, 
                                args=(batch_id, _process_name, 
                                      output_dict[_student_name][0] if _need_logits else None,
                                      output_dict[_student_name][1] if need_attn_v else None, 
                                      output_dict[_student_name][2] if need_attn_v else None))
                logging.debug(f"teacher_inference, send after student ready, batch_id: {batch_id}, shape: {output_dict[_student_name][0].shape}, {output_dict[_student_name][1].shape}, {output_dict[_student_name][2].shape}")
                rpc_trans_finish.wait() # wait for student to receive the result
                output_dict.pop(_student_name)
                fut_trans_finish.set_result(True)
            return callback_student_ready
        
        tmp_fut_trans_finish = torch.futures.Future()
        fut_ready.add_done_callback( # fut_ready will be resolved if student is ready to receive date
            wrap_callback(name, tmp_fut_trans_finish)
        )
        future_wait_students_receive.append(tmp_fut_trans_finish)

    # if all results are sent, and student process the data, then this function is finished
    # then teacher can compute the next batch
    torch.futures.wait_all(future_wait_students_receive)

    logging.debug(f"teacher_inference, batch_id: {batch_id}, finish")




###### student worker ########

_student_model = None
_opt = None
_scheduler = None
_train_config: Distill_train_config = None


def student_worker_main(process_id, 
                process_name, 
                ddp_world_size, rpc_world_size, 
                student_ddp_port, student_rpc_port,
                process_name_and_device_map: list, 
                student_model, 
                train_config: Distill_train_config, 
                training_name: str, 
                logger_dir: str,
                logger_name: str):
    global _process_name
    global _worker_device_name
    global _student_model
    global _opt
    global _training_name
    global _db_logger
    global _log_dir
    global _log_name
    _training_name = training_name
    _log_dir = logger_dir
    _log_name = logger_name
    _db_logger = DB_logger(_log_name, _log_dir)

    logging.debug(f"student_worker_main, process_name: {process_name}, start init, pid: {process_id}, ddp_world_size: {ddp_world_size}, rpc_world_size: {rpc_world_size}, student_ddp_port: {student_ddp_port}, student_rpc_port: {student_rpc_port}")

    _process_name = process_name
    ip = get_localhost_ip()
    worker_device_name = process_name_and_device_map[process_name]
    _worker_device_name = worker_device_name

    gpu_id = None
    if worker_device_name.startswith('cuda'):
        gpu_id = int(worker_device_name.split(':')[1])
    
    student_setup_training(student_model, worker_device_name, train_config)
    
    # create rpc channel config
    rpc_backend_config = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=8,
        rpc_timeout=500,
        init_method=f'tcp://{ip}:{student_rpc_port}',
    )
    if gpu_id != None: # this step setup the connection between each worker's gpu, so rpc call can directly send tensor to other worker's gpu
        for name in process_name_and_device_map:
            if name == process_name:
                continue
            device_name = process_name_and_device_map[name]
            # if device_name starts with cuda, then it is a gpu device
            if device_name.startswith('cuda'):
                rpc_backend_config.set_device_map(name, {gpu_id: int(device_name.split(':')[1])})
        rpc_backend_config.set_devices([f'cuda:{gpu_id}'])
    
    logging.debug(f"student_worker_main, process_name: {process_name}, pid: {process_id}, \
                  world_size: {rpc_world_size}, {ddp_world_size}, gpu_id: {gpu_id}, device_name: {worker_device_name}")


    logging.debug(f"student_worker_main, process_name: {process_name}, before init ddp")
    if ddp_world_size > 1:
        # init process group for ddp
        student_id = int(_process_name.split('_')[-1])
        dist.init_process_group(backend='nccl',
                                init_method=f'tcp://{ip}:{student_ddp_port}',
                                world_size=ddp_world_size,
                                rank=student_id)
        _student_model = torch.nn.parallel.DistributedDataParallel(_student_model, device_ids=[gpu_id] if gpu_id != None else None)
    
    logging.debug(f"student_worker_main, process_name: {process_name}, finish init ddp")

    # start rpc
    rpc.init_rpc(process_name,
                 rank=process_id, 
                 world_size=rpc_world_size, 
                 rpc_backend_options=rpc_backend_config)
    


    logging.info(f"student_worker_main, process_name: {process_name}, finish init")

    # wait for rpc call
    global _clog_end
    _clog_end = torch.futures.Future()
    _clog_end.wait()

    rpc.shutdown()
    if ddp_world_size > 1:
        dist.destroy_process_group()
    logging.info(f"student_worker_main, process_name: {process_name}, exit")


def student_setup_training(
        student_model: nn.Module, worker_device_name,
        train_config: Distill_train_config
):
    global _student_model
    global _opt
    global _scheduler
    global _train_config
    global _need_attn_v
    global _need_logits

    _train_config = train_config
    if _train_config.soft_loss_weight == 0:
        _need_logits = False
        logging.warn(f"student_setup_training, soft_loss_weight is 0, so logits is not needed, set _need_logits to False")
    if _train_config.mimic_loss_weight == 0:
        _need_attn_v = False
        logging.warn(f"student_setup_training, mimic_loss_weight is 0, so attn loss is not needed, set _need_attn_v to False")

    # setup model
    if student_model is None:
        student_model = _student_model
    _student_model = student_model.to(worker_device_name)
    del student_model

    opt_name = "adamw"
    if hasattr(_train_config, "opt_name") and _train_config.opt_name != None:
        opt_name = _train_config.opt_name
    if opt_name == "sophia":
        logging.info(f"student_setup_training, use SophiaG")
        _opt = SophiaG(_student_model.parameters(),
                        lr=_train_config.lr,
                        weight_decay=_train_config.weight_decay, 
                        betas=(_train_config.beta1, _train_config.beta2),
                        rho=0.03)
    elif opt_name == "lion":
        logging.info(f"student_setup_training, use Lion")
        _opt = Lion(_student_model.parameters(),
                    lr = _train_config.lr,
                    betas = (_train_config.beta1, _train_config.beta2),
                    weight_decay = _train_config.weight_decay,
                    use_triton=False)
    elif opt_name == "adamw":
        _opt = torch.optim.AdamW(_student_model.parameters(), 
                                    lr=_train_config.lr,
                                    weight_decay=_train_config.weight_decay, 
                                    betas=(_train_config.beta1, _train_config.beta2),
                                    eps=_train_config.eps)
    else:
        raise RuntimeError(f"student_setup_training, unknown opt_name: {opt_name}")
    
    #_opt.zero_grad()

    if _train_config.slr_seg != None:
        _scheduler = Linear_schedular_seg(_opt, _train_config.slr_seg)
    else:
        _scheduler = Linear_schedular_with_warmup(_opt, 
                                            warmup_start_factor=_train_config.llr_warmup_start_factor,
                                            start_factor=_train_config.llr_start_factor,
                                            end_factor=_train_config.llr_end_factor,
                                            warmup_iters=_train_config.llr_warmup_iters,
                                            total_iters=_train_config.llr_total_iters,
                                            last_epoch=_train_config.llr_last_epoch)
    
    if _train_config.ckpt_path != None:
        states = torch.load(_train_config.ckpt_path, map_location=worker_device_name)

        model_states = states["model"]
        unwanted_prefix_dueto_compile = '_orig_mod.'
        unwanted_prefix_dueto_ddp = 'module.'
        unwanted_prefix_dueto_ddp_compiled = 'module._orig_mod.'

        for k,v in list(model_states.items()):
            if k.startswith(unwanted_prefix_dueto_ddp_compiled):
                model_states[k[len(unwanted_prefix_dueto_ddp_compiled):]] = model_states.pop(k)
            elif k.startswith(unwanted_prefix_dueto_ddp):
                model_states[k[len(unwanted_prefix_dueto_ddp):]] = model_states.pop(k)
            elif k.startswith(unwanted_prefix_dueto_compile):
                model_states[k[len(unwanted_prefix_dueto_compile):]] = model_states.pop(k)

        logging.info(f"student_setup_training, load model from ckpt")
        _student_model.load_state_dict(model_states)
        if not _train_config.ckpt_ignore_opt:
            logging.info(f"student_setup_training, load opt and scheduler from ckpt")
            _opt.load_state_dict(states['opt'])
            _scheduler.load_state_dict(states['scheduler'])

    if _compile_model == True:
        _student_model = torch.compile(_student_model)



_teacher_future = None # list, [(name, future), ...], if student does not get the result after model forward, it will wait for the result
_teacher_result = None # dict, key: teacher_name, value: (logits, attn, v)
_current_receiving_teacher_batch_id = -1 # -1 means no new batch is receiveds, otherwise it will be a valid batch id
_lock_current_receiving_teacher_batch_id = threading.Lock()
_queue_waiting_ready_for_transfer = []

_need_attn_v = True
_need_logits = True


def _trigger_ready_for_transfer():
    global _queue_waiting_ready_for_transfer
    global _current_batch_id
    logging.debug(f"student: {_process_name}: _trigger_ready_for_transfer() is called, will release {len(_queue_waiting_ready_for_transfer)} futures")
    new_queue = []
    for batch_id, fut in _queue_waiting_ready_for_transfer:
        if batch_id <= _current_batch_id + 1:
            fut.set_result(True)
        else:
            new_queue.append((batch_id, fut))
    _queue_waiting_ready_for_transfer = new_queue
    if len(_queue_waiting_ready_for_transfer) > 0:
        logging.warn(f"""student: {_process_name}: _trigger_ready_for_transfer() is called, \
                     but there are still {len(_queue_waiting_ready_for_transfer)} futures not released, \
                     which means the teacher is at least 2 batch beyond the current student batch_id. \n\
                     {_queue_waiting_ready_for_transfer}, \n\
                     current_batch_id: {_current_batch_id}""")
        

def receive_teacher_result(batch_id, t_name, t_logits, t_attn, t_v): # should called by the teacher, transfer the result to the student
    # after this function is called, even if the student is not ready to receive the resulte, 
    # those tensor already arrived at the student's gpu
    global _teacher_future
    global _teacher_result
    global _current_receiving_teacher_batch_id

    logging.debug(f"student: {_process_name}: receive_teacher_result() is called for batch {batch_id}, from teacher: {t_name}")
    try:
        logging.debug(f"student: {_process_name}: receive_teacher_result(): any nan: {torch.isnan(t_logits).any()}, {torch.isnan(t_attn).any()}, {torch.isnan(t_v).any()}")
    except:
        pass
    with _lock_current_receiving_teacher_batch_id:
        if not (_current_receiving_teacher_batch_id == batch_id or _current_receiving_teacher_batch_id == -1):
            raise RuntimeError(f"student: {_process_name}: receive_teacher_result() is called for batch {batch_id}, \
                             but the current receiving batch is {_current_receiving_teacher_batch_id}, \
                            which means the previous train() is not finished yet.")
        
        if _teacher_future == None:
            _teacher_future = []
        if _teacher_result == None:
            _teacher_result = {}
        if _current_receiving_teacher_batch_id == -1:
            _current_receiving_teacher_batch_id = batch_id

        for i in range(len(_teacher_future)): # to see any future is waiting for this teacher's result
            if _teacher_future[i][0] == t_name:
                _teacher_future[i][1].set_result(True)
                break
        _teacher_result[t_name] = (t_logits, t_attn, t_v) # should sent to the gpu of the student worker

    logging.debug(f"student: {_process_name}: receive_teacher_result() is finished for batch {batch_id}")


def until_ready_for_transfer(batch_id, teacher_name):
    global _current_receiving_teacher_batch_id
    global _lock_current_receiving_teacher_batch_id
    logging.debug(f"student: {_process_name}: until_ready_for_transfer() is called for batch {batch_id}")

    fut_ready_recv_teacher_result = None

    with _lock_current_receiving_teacher_batch_id:
        if not (_current_receiving_teacher_batch_id == -1 or _current_receiving_teacher_batch_id == batch_id):
            logging.debug(f"student: {_process_name}: arrive too early, until_ready_for_transfer() is waiting for batch {batch_id}")
            fut_ready_recv_teacher_result = torch.futures.Future()
            global _queue_waiting_ready_for_transfer
            _queue_waiting_ready_for_transfer.append((batch_id, fut_ready_recv_teacher_result))

    if fut_ready_recv_teacher_result != None:
        fut_ready_recv_teacher_result.wait()
    
    logging.debug(f"student: {_process_name}: until_ready_for_transfer() is finished for batch {batch_id}")
    return (_need_attn_v, _need_logits)



def student_train(
        batch_id: int, 
        input_ids: torch.Tensor,
        pred_ids: torch.Tensor,
        loss_factor: float, # scale down the loss, because the batch is composed by multiple student and mini-batch
        need_step: bool,
        teacher_names: list, 
        current_step
):
    global _student_model
    global _opt
    global _worker_device_name
    global _teacher_future
    global _teacher_result
    global _current_batch_id
    global _current_receiving_teacher_batch_id
    global _lock_current_receiving_teacher_batch_id
    global _train_config
    global _need_logits
    global _need_attn_v
    global _current_step

    # prevent multiple training call when the previous training is not finished
    if _current_batch_id == batch_id:
        raise RuntimeError(f"student: {_process_name}: train() is called multiple times for the same batch {batch_id}")
    
    _current_step = current_step
    _current_batch_id = batch_id
    logging.debug(f"student: {_process_name}: train() is called for batch {batch_id}")
    _student_model.train()
    input_ids = input_ids.to(_worker_device_name)
    pred_ids = pred_ids.to(_worker_device_name)
    
    train_start_time = time.time()
    s_logits, s_attn, s_v = get_student_output(_student_model, input_ids)
    s_attn = distill_util.post_process_attn(s_attn)
    s_v = distill_util.post_process_v(s_v)

    # wait for teacher results
    wait_start_time = time.time()
    with _lock_current_receiving_teacher_batch_id:
        # if teacher's result does not arrive, wait for it
        _teacher_future = [] # student need teacher results to compute loss
        _teacher_future_no_name = []
        for t_name in teacher_names:
            if _teacher_result == None or t_name not in _teacher_result: # it means this teacher's result is not received yet
                tmp_fut = torch.futures.Future()
                new_future = (t_name, tmp_fut)
                _teacher_future.append(new_future)
                _teacher_future_no_name.append(tmp_fut)
    if len(_teacher_future_no_name) > 0:
        torch.futures.wait_all(_teacher_future_no_name)
    wait_end_time = time.time()
    if _process_name == "student_0" and batch_id % 100 == 0:
        _db_logger.log(NAME_STUDENT_WAIT_TIME_1_ITER, wait_end_time - wait_start_time, batch_id)

    # merge teacher results; NO NEW RESULT SHOULD BE RECEIVED FROM THIS POINT, prevent cuda out of memory
    with _lock_current_receiving_teacher_batch_id:
        t_logits = []
        t_attn = []
        t_v = []
        for name in teacher_names:
            if name not in _teacher_result:
                raise RuntimeError(f"student: {_process_name}: train() is called for batch {batch_id}, \
                                but the teacher {name}'s result is not received")
            logging.debug(f"student, concat teacher result; batch_id: {batch_id}, name: {name}")
            try:
                logging.debug(f"shape: {_teacher_result[name][0].shape}, {_teacher_result[name][1].shape}, {_teacher_result[name][2].shape}")
                logging.debug(f"student, concat teacher result; any nan: {torch.isnan(_teacher_result[name][0]).any()}, {torch.isnan(_teacher_result[name][1]).any()}, {torch.isnan(_teacher_result[name][2]).any()}")
            except:
                pass
            t_logits.append(_teacher_result[name][0])
            t_attn.append(_teacher_result[name][1])
            t_v.append(_teacher_result[name][2])
    if _need_logits:
        t_logits = torch.cat(t_logits, dim=0).detach()
    if _need_attn_v:
        t_attn = torch.cat(t_attn, dim=0).detach()
        t_v = torch.cat(t_v, dim=0).detach()

    try:
        if torch.isnan(t_logits).any() or torch.isnan(t_attn).any() or torch.isnan(t_v).any():
            logging.error(f"student train: after concat teacher results, there is nan in the result, batch_id: {batch_id}")
            logging.error(f"{torch.isnan(_teacher_result[name][0]).any()}, {torch.isnan(_teacher_result[name][1]).any()}, {torch.isnan(_teacher_result[name][2]).any()}")
    except:
        pass

    _teacher_future = None
    _teacher_result = None

    loss_h = loss_hard_label(s_logits, input_ids, pred_ids)
    
    if _train_config.soft_loss_weight != 0:
        loss_s = loss_soft_logits(distill_util.post_process_logits(s_logits), t_logits, temperature=_train_config.loss_soft_temperature)
    else:
        loss_s = torch.tensor(0.0, device=_worker_device_name)
    if _train_config.mimic_loss_weight != 0:
        student_sliding_window_size = input_ids.shape[1]
        if hasattr(_student_model, "dearth_config"):
            student_sliding_window_size = _student_model.dearth_config.sliding_window_size
        else:
            student_sliding_window_size = _student_model.module.dearth_config.sliding_window_size
        loss_m = loss_mimic_attn(s_attn, s_v, t_attn, t_v, virtual_v_head_num=_train_config.virtual_v_head_num, sliding_window_size=student_sliding_window_size)
    else:
        loss_m = torch.tensor(0.0, device=_worker_device_name)

    loss = _train_config.hard_loss_weight * loss_h + \
              _train_config.soft_loss_weight * loss_s + \
                _train_config.mimic_loss_weight * loss_m
    loss = loss * loss_factor
    # if loss is nan, warning and skip this batch
    if torch.isnan(loss).any():
        logging.warning(f"student: {_process_name}: train() get nan loss for batch {batch_id}, skip this batch")
        _current_batch_id = -1
        with _lock_current_receiving_teacher_batch_id:
            _current_receiving_teacher_batch_id = -1 # -1 means no new batch is receiveds, otherwise it will be a valid batch id
            # no one should modifify the _queue_waiting_ready_for_transfer for now. 
            _trigger_ready_for_transfer()
        return (loss.item(), loss_h.item() * loss_factor, loss_s.item() * loss_factor, loss_m.item() * loss_factor)
    loss.backward()

    del s_logits, s_attn, s_v, t_logits, t_attn, t_v

    train_end_time = time.time()
    if _process_name == "student_0" and batch_id % 100 == 50:
        _db_logger.log(NAME_STUDENT_TRAIN_TIME_1_ITER, train_end_time - train_start_time, batch_id)
        logging.info(f"student: {_process_name}: train() time for 1 iter: {train_end_time - train_start_time}, batch_id: {batch_id}")
    
    # ready to receive the next batch's teacher result, because backward should release the graph and current batch's result
    with _lock_current_receiving_teacher_batch_id:
        _current_receiving_teacher_batch_id = -1 # -1 means no new batch is receiveds, otherwise it will be a valid batch id
        # no one should modifify the _queue_waiting_ready_for_transfer for now. 
        _trigger_ready_for_transfer()
    # ALLOW next teacher result from this point

    if need_step:
        # apply gradient clipping
        torch.nn.utils.clip_grad_value_(parameters=_student_model.parameters(), clip_value=_train_config.gradient_clip)
        logging.info(f"student: {_process_name}: batch {batch_id}, current lr: {_scheduler.get_last_lr()}")
        if _process_name == "student_0":
            _db_logger.log(NAME_LR, _scheduler.get_last_lr()[0], _current_step)
        _opt.step()
        _opt.zero_grad()
        _scheduler.step()
    
    _current_batch_id = -1
    logging.debug(f"student: {_process_name}: train() is finished for batch {batch_id}")

    return (loss.item(), loss_h.item() * loss_factor, loss_s.item() * loss_factor, loss_m.item() * loss_factor)


def _sync_model_param():
    global _student_model
    if dist.is_initialized():
        for p in _student_model.parameters():
            dist.broadcast(p, src=0)

def _student_save_model(file_path, step):
    global _student_model
    # save as bf16
    model_states = _student_model.state_dict()
    for k in model_states:
        model_states[k] = model_states[k].bfloat16()
    save_dict = {
        "model": model_states,
        "opt": _opt.state_dict(),
        "scheduler": _scheduler.state_dict(),
        "step": step
    }
    torch.save(save_dict, file_path)
    logging.info(f"student: {_process_name}: save model to {file_path}")