import torch
import multiprocessing as mp

import lmdb
import lz4.frame
import os
import logging
import numpy as np
import time
import math

from dblogger import DB_logger


class Dataloader:
    def __init__(self, data_rootpath: str, tokenizer, config: object, seqlen: int, worker_num=None, log_dir=None, log_name=None):
        # check data_rootpath exist and is a directory
        if not os.path.exists(data_rootpath):
            raise RuntimeError(f"data_rootpath {data_rootpath} does not exist")
        if not os.path.isdir(data_rootpath):
            raise RuntimeError(f"data_rootpath {data_rootpath} is not a directory")
        
        if worker_num is None:
            worker_num = max(int(mp.cpu_count()*0.6), 8)

        self.pipe_to_manager, p = mp.Pipe()
        self.manager = mp.Process(target=_manager_main, 
            args=(data_rootpath, tokenizer, config, p, seqlen, worker_num, log_dir, log_name))
        self.manager.start()

        self.is_alive = True

    def get_batch(self, batch_size, seqlen=None) -> torch.Tensor:
        self.pipe_to_manager.send({
            "task_name": "get",
            "bz": batch_size,
            "seqlen": seqlen,
            "batch_id": 0,
        })
        ret = self.pipe_to_manager.recv()
        return ret

    def destory(self):
        self.__del__()

    def __del__(self):
        if not self.is_alive:
            return
        logging.debug("Dataloader: __del__")
        self.pipe_to_manager.send({
            "task_name": "exit"
        })
        self.manager.join()
        self.pipe_to_manager.close()
        logging.debug("Dataloader: __del__ finish")
        self.is_alive = False

def _manager_main(data_rootpath: str, tokenizer, config: dict, pipe: mp.Pipe, seqlen: int, worker_num=4, log_dir=None, log_name=None):
    manager = Dataloader_manager(data_rootpath, tokenizer, config, pipe, seqlen, worker_num, log_dir, log_name)
    manager.run()


class Dataloader_manager:
    def __init__(self, data_rootpath: str, tokenizer, config: dict, pipe: mp.Pipe, seqlen: int, worker_num=4, log_dir=None, log_name=None):
        self.worker_num = worker_num
        self.data_rootpath = data_rootpath
        self.task_queue = mp.Queue()
        self.workers = []
        self.pipes_to_worker = []

        self.pipe_to_master = pipe

        self.dataset_names: list = list(config.keys())
        self.dataset_config = config
        self.dataset_weight_sum = 0
        self.check_and_compute_dataset()

        self.MIN_TOKENS_CNT = 10000
        self.tokens_buf = []
        self.waiting_tasks = {}
        self.batch_tokens_buf = {} # seqlen -> [tokens]
        # self.tmp_tokens_buf = [] # for fill_batch_tokens_buf()

        self.recent_seqlen = seqlen
        self.cnt_tasks_send = 0

        for i in range(worker_num):
            pipe1, pipe2 = mp.Pipe()
            self.pipes_to_worker.append(pipe1)
            self.workers.append(mp.Process(target=_worker_main, args=(data_rootpath, self.dataset_names,
                                                                      tokenizer, i, 
                                                                      self.task_queue, pipe2,
                                                                      log_dir, log_name)))
        
        for worker in self.workers:
            worker.start()
    
    def check_and_compute_dataset(self):
        for name in self.dataset_names:
            dataset_path = os.path.join(self.data_rootpath, name)
            if not os.path.exists(dataset_path):
                logging.error(f"dataset {name} does not exist")
                self.dataset_names.remove(name)
                continue
            if not os.path.isdir(dataset_path):
                logging.error(f"dataset {name} is not a directory")
                self.dataset_names.remove(name)
                continue
            assert self.dataset_config[name] > 0, f"dataset {name} has weight <= 0"
            self.dataset_weight_sum += self.dataset_config[name]
        if len(self.dataset_names) == 0:
            raise RuntimeError("no dataset to load")
        self.dataset_prob = []
        for name in self.dataset_names:
            self.dataset_prob.append(self.dataset_config[name] / self.dataset_weight_sum)
        self.dataset_prob = np.array(self.dataset_prob)

    def run(self):
        if len(self.dataset_names) == 0:
            raise RuntimeError("no dataset to load")
        need_data_info = None # (bz, seqlen, batch_id)
        while True:
            if need_data_info is not None:
                required_seqlen = need_data_info[1]
                required_bz = need_data_info[0]
                batch_id = need_data_info[2]
                if self.batch_tokens_buf[required_seqlen] is not None \
                    and len(self.batch_tokens_buf[required_seqlen]) >= required_bz:
                    # if we have enough data for this seqlen, send it
                    ret = self.batch_tokens_buf[required_seqlen][:required_bz]
                    self.batch_tokens_buf[required_seqlen] = self.batch_tokens_buf[required_seqlen][required_bz:]
                    ret = torch.tensor(ret, dtype=torch.long)
                    self.pipe_to_master.send(ret)
                    need_data_info = None
                else:
                    # if we don't have enough data for this seqlen, fetch more
                    logging.warn(f"not enough data for seqlen {required_seqlen}, current size {len(self.batch_tokens_buf[required_seqlen])}, token_buf size {len(self.tokens_buf)}")
                    self.fill_batch_tokens_buf(required_seqlen, buf_size=required_bz*2)
                    self.fetch_more()
            if self.pipe_to_master.poll():
                cmd = self.pipe_to_master.recv()
                if cmd["task_name"] == "get":
                    batch_id = cmd["batch_id"]
                    required_seqlen = None if "seqlen" not in cmd else cmd["seqlen"]
                    required_bz = cmd["bz"]
                    if required_seqlen is None:
                        required_seqlen = self.recent_seqlen
                    else:
                        self.recent_seqlen = required_seqlen
                    need_data_info = (required_bz, required_seqlen, batch_id)
                elif cmd["task_name"] == "exit":
                    break
                else:
                    raise RuntimeError(f"Dataloader_manager: unknown command {cmd}")
            self.dump_pipe()
            self.fill_batch_tokens_buf()
            self.fetch_more()
            time.sleep(0.01)
    

    def dump_pipe(self):
        recv_something = False
        for pipe in self.pipes_to_worker:
            if pipe.poll():
                tmp = pipe.recv()
                if tmp is None:
                    continue
                recv_something = True
                task_id = tmp["task_id"]
                result = tmp["result"]
                if result is None:
                    continue
                self.tokens_buf.extend(result)
                if task_id in self.waiting_tasks:
                    #del self.waiting_tasks[task_id]
                    self.waiting_tasks.pop(task_id)
        if recv_something:
            np.random.shuffle(self.tokens_buf)
        

    def fill_batch_tokens_buf(self, required_seqlen=None, buf_size=2500):
        if required_seqlen is None:
            required_seqlen = self.recent_seqlen
        if required_seqlen not in self.batch_tokens_buf:
            self.batch_tokens_buf[required_seqlen] = []
        
        cnt_new_seq_added = 0
        MAX_BATCH_BUF_SIZE = max(2500, buf_size)
        tmp_tokens_buf = []

        while len(self.batch_tokens_buf[required_seqlen]) < MAX_BATCH_BUF_SIZE \
            and len(self.tokens_buf) > 5:

            sample = self.tokens_buf.pop(-1)
            sample_seqlen = len(sample)
            if sample_seqlen > required_seqlen:
                tmp_tokens_buf.extend(sample)
                seg_count = math.ceil(sample_seqlen / required_seqlen)
                if sample_seqlen % required_seqlen != 0:
                    while len(tmp_tokens_buf) < seg_count * required_seqlen:
                        more_sample = self.tokens_buf.pop(-1)
                        tmp_tokens_buf.extend(more_sample)
                for i in range(seg_count):
                    self.batch_tokens_buf[required_seqlen].append(tmp_tokens_buf[i*required_seqlen:(i+1)*required_seqlen])
                cnt_new_seq_added += seg_count
                # include the last part of the sample, but may ignore tokens from the more_sample
                tmp_tokens_buf = []
                    
            elif sample_seqlen == required_seqlen:
                self.batch_tokens_buf[required_seqlen].append(sample)
                cnt_new_seq_added += 1
                tmp_tokens_buf = []
            else:
                tmp_tokens_buf.extend(sample)
                while len(tmp_tokens_buf) < required_seqlen:
                    more_sample = self.tokens_buf.pop(-1)
                    tmp_tokens_buf.extend(more_sample)
                self.batch_tokens_buf[required_seqlen].append(tmp_tokens_buf[:required_seqlen])
                cnt_new_seq_added += 1
                tmp_tokens_buf = []
                # throw any extra tokens
            # this function make most of the sample start with <s>, unless the sample is extremely long;
            # it may make the distribution more similar to the inference input, because the inference will always start with <s>, as the first token


            
        # shuffle the buf with required_seqlen
        np.random.shuffle(self.batch_tokens_buf[required_seqlen])
        return cnt_new_seq_added


    def fetch_more(self):
        sample_needed = self.MIN_TOKENS_CNT - len(self.tokens_buf)
        if not (sample_needed > 0.3 * self.MIN_TOKENS_CNT):
            return
        
        sample_needed = min(sample_needed, 8000)
        sample_needed = max(sample_needed, 2000)

        current_time = time.time()
        # count waiting tasks within recent 30 seconds
        
        cnt_waiting_tasks = 0
        for task_id in self.waiting_tasks:
            task_time = self.waiting_tasks[task_id]
            if current_time - task_time < 30:
                cnt_waiting_tasks += 1
            else:
                if task_id in self.waiting_tasks:
                    self.waiting_tasks.pop(task_id)
        
        # if there are too many waiting tasks, we don't fetch more data
        if cnt_waiting_tasks > 30:
            return
        
        MAX_SIMPLE_EACH_TASK = 200

        sample_idxs = np.random.choice(a=len(self.dataset_names), size=sample_needed, p=self.dataset_prob)

        # fetch more data
        for i, dataset_name in enumerate(self.dataset_names):
            num_samples_total = int(np.sum(sample_idxs == i))
            logging.debug(f"fetching {num_samples_total} samples from {dataset_name}, {num_samples_total / sample_needed * 100:.2f}%")
            while num_samples_total > 0:
                num_samples = min(num_samples_total, MAX_SIMPLE_EACH_TASK)
                num_samples_total -= num_samples
                task_id = self.cnt_tasks_send
                self.cnt_tasks_send += 1
                task = {
                    "task_name": "get",
                    "task_id": task_id,
                    "db_name": dataset_name,
                    "num_samples": num_samples,
                }
                self.task_queue.put(task, block=False)
                self.waiting_tasks[task_id] = time.time()

    def __del__(self):
        logging.debug("Dataloader_manager: __del__")
        for pipe in self.pipes_to_worker:
            pipe.send("exit")
        for worker in self.workers:
            worker.join()
        self.task_queue.close()
        self.pipe_to_master.close()
        for pipe in self.pipes_to_worker:
            pipe.close()
        logging.debug("Dataloader_manager: __del__ finish")

def _worker_main(data_rootpath: str, db_names: list, 
                 tokenizer, worker_id: int, task_queue, pipe, 
                 log_dir, log_name):
    worker = Dataloader_worker(data_rootpath, db_names, tokenizer, worker_id, task_queue, pipe, log_dir, log_name)
    worker.run()

class Dataloader_worker:
    def __init__(self, data_rootpath: str, db_names: list, 
                 tokenizer, worker_id: int, 
                 task_queue: mp.Queue, pipe: mp.Pipe,
                 log_dir, log_name):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.pipe = pipe

        self.tokenizer = tokenizer
        self.data_rootpath = data_rootpath
        self.db_names = db_names
        self.task_finished_cnt = 0

        self.db_logger = None
        if self.worker_id == 0 and log_dir is not None and log_name is not None:
            self.db_logger = DB_logger(logger_name=log_name, log_dir=log_dir)
            logging.info(f"Dataloader_worker {self.worker_id} will log to {log_dir}/{log_name}")

        self.lmdb_envs = {}
        self.lmdb_start_idx = {}
        self.lmdb_end_idx = {}
        self.open_lmdbs(data_rootpath, db_names)


    def open_lmdbs(self, data_rootpath, db_names):
        for name in db_names:
            db_path = os.path.join(data_rootpath, name)
            try:
                self.lmdb_envs[name] = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
                self.lmdb_start_idx[name] = 0
                self.lmdb_end_idx[name] = self.lmdb_envs[name].stat()["entries"]
                if self.worker_id == 0:
                    logging.info(f"lmdb {name} has {self.lmdb_end_idx[name]} samples")
            except:
                logging.error(f"failed to open lmdb {name}")

    def close_lmdbs(self):
        if self.lmdb_envs is None:
            return
        for name in self.lmdb_envs:
            try:
                self.lmdb_envs[name].close()
            except:
                logging.error(f"failed to close lmdb {name}")
    
    def run(self):
        while True:
            task = None
            try:
                task = self.task_queue.get(block=False)
            except:
                pass

            cmd = None
            if self.pipe.poll():
                cmd = self.pipe.recv()
            
            if cmd == "exit":
                break
            elif task != None and task["task_name"] == "get":
                task_id = task["task_id"]
                db_name = task["db_name"]
                task_num_samples = task["num_samples"]

                ret = self.get_data(db_name, task_num_samples)

                dict_to_send = {
                    "task_id": task_id,
                    "result": ret
                }
                self.pipe.send(dict_to_send)
                self.task_finished_cnt += 1
            elif task != None:
                logging.error(f"unknown task {task}")
                break
            time.sleep(0.05)

    def get_data(self, db_name, num_samples):
        db_env = self.lmdb_envs[db_name]
        db_start_idx = self.lmdb_start_idx[db_name]
        db_end_idx = self.lmdb_end_idx[db_name]
        ret = []
        with db_env.begin(write=False) as txn:
            rand_idxs = np.random.randint(db_start_idx, db_end_idx, size=num_samples)
            rand_idxs = np.sort(rand_idxs)
            if self.db_logger is not None:
                self.db_logger.log("data_idx", rand_idxs, self.task_finished_cnt, "sampling-info")
            for i in range(num_samples):
                rand_idx = int(rand_idxs[i])
                text = txn.get(str(rand_idx).encode('utf-8'))
                if text is None:
                    continue
                text = lz4.frame.decompress(text)
                text = text.decode('utf-8')
                tokens = self.tokenizer.encode(text) # TODO: change here if tokenizer is changed
                tokens = post_tokenize_process(tokens, self.tokenizer)
                if self.db_logger is not None and self.task_finished_cnt % 10 == 0 and i == 0:
                    self.db_logger.log("tokens", tokens, self.task_finished_cnt, "sampling-info")
                ret.append(tokens)
        return ret

    def __del__(self):
        if self is None:
            return
        if not hasattr(self, "worker_id"):
            return
        self.close_lmdbs()
        self.pipe.close()
        self.task_queue.close()
        logging.info(f"worker {self.worker_id} is exiting")


def post_tokenize_process(token_ids, tokenizer) -> list:
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    ret = [bos_id] # 1 is the token_id of <s>

    i = 0
    for token_id in token_ids:
        if i == 0 and token_id == bos_id:
            i += 1
            continue
        if token_id == bos_id: # 1 is the token_id of <s>
            ret.append(tokenizer.get_vocab()["<"])
            ret.append(tokenizer.get_vocab()["s"])
            ret.append(tokenizer.get_vocab()[">"])
            logging.warn(f"token_id 1 appears in the middle of the text, i={i}")
        elif token_id == eos_id: # 2 is the token_id of </s>
            ret.append(tokenizer.get_vocab()["<"])
            ret.append(tokenizer.get_vocab()["/"])
            ret.append(tokenizer.get_vocab()["s"])
            ret.append(tokenizer.get_vocab()[">"])
            logging.warn(f"token_id 2 appears in the middle of the text, i={i}")
        else:
            ret.append(token_id)
        i += 1
    
    ret.append(eos_id) # 2 is the token_id of </s>
    
    return ret
