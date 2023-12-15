import torch
import math
import logging

def get_localhost_ip():
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return ip


def generate_names_device(n_teacher, n_student):
    available_device = torch.cuda.device_count()
    assert n_teacher + n_student <= available_device, f"n_teacher + n_student = {n_teacher + n_student} > {available_device} = available_device, all models must on different cuda device"
    rank_cnt = 1 # rank 0 is reserved for main process
    dev_n = 0
    ret = []
    for i in range(n_teacher):
        t_name = "teacher_" + str(i)
        t_dev = "cuda:" + str(dev_n)
        ret += [(t_name, t_dev, rank_cnt)]
        rank_cnt += 1
        dev_n += 1

    for i in range(n_student):
        s_name = "student_" + str(i)
        s_dev = "cuda:" + str(dev_n)
        ret += [(s_name, s_dev, rank_cnt)]
        rank_cnt += 1
        dev_n += 1

    process_name_and_device_map = {}
    for name, dev, rank in ret:
        process_name_and_device_map[name] = dev

    student_names_and_id = {}
    teacher_names_and_id = {}
    for name, dev, rank in ret:
        if name.startswith("teacher"):
            teacher_names_and_id[name] = rank
        else:
            student_names_and_id[name] = rank

    return teacher_names_and_id, student_names_and_id, process_name_and_device_map


def split_batch_config(
    batch_size,
    teacher_names: list,
    student_names: list,
):
    '''
    return: teacher_send_config, student_recv_config
        the two config is used to split batch
        teacher_send_config: {key: teacher_name, value: list[(s_name, (start_idx, end_idx))]}
        student_recv_config: {key: student_name, value: list[t_name]}
    '''
    n_student = len(student_names)
    n_teacher = len(teacher_names)
    assert batch_size % n_student == 0, f"{batch_size} % {n_student} != 0"
    if n_teacher != 0:
        assert batch_size % n_teacher == 0, f"{batch_size} % {n_teacher} != 0"
    batch_per_student = batch_size // n_student
    batch_per_teacher = batch_size // max(n_teacher, 1)

    teacher_send_config = {} # teacher_name -> list[(s_name, (start_idx, end_idx))]
    student_recv_config = {} # student_name -> list[t_name]

    for s_names in student_names:
        student_recv_config[s_names] = []

    s_i = 0
    s_remain = batch_per_student
    teacher_batch_start_idx = 0
    for t_i, t_name in enumerate(teacher_names):
        assigned_cnt = 0
        tmp_t_config = []
        while assigned_cnt < batch_per_teacher:
            s_name = student_names[s_i]
            start_idx = assigned_cnt
            end_idx = start_idx + s_remain
            if end_idx > batch_per_teacher:
                end_idx = batch_per_teacher
            s_remain -= (end_idx - start_idx)

            if s_name not in student_recv_config:
                student_recv_config[s_name] = []
            student_recv_config[s_name].append(t_name)
            tmp_t_config.append((s_name, (start_idx, end_idx)))
            assigned_cnt += (end_idx - start_idx)

            if s_remain == 0:
                s_i += 1
                s_remain = batch_per_student
        teacher_send_config[t_name] = tmp_t_config
        teacher_batch_start_idx += batch_per_teacher

    return teacher_send_config, student_recv_config


def split_betch(batch, names):
    ret = {}
    batch_size = batch.shape[0]
    names_cnt = len(names)
    assert batch_size % names_cnt == 0, f"{batch_size} % {names_cnt} != 0"
    batch_per_worker = batch_size // names_cnt
    start_idx = 0
    for name in names:
        ret[name] = batch[start_idx:start_idx + batch_per_worker]
        start_idx += batch_per_worker
    return ret



def get_teacher_output(teacher, token_ids):
    '''
    teacher: a pretrained model
    token_ids: a tensor of token ids
    teacher_target_layer: the layer of the teacher model to use as the target, 1-based

    returns: (t_logits, t_attn, t_v)
        if used multi-query, q and k will ensure to have same shape
    '''
    token_ids = token_ids.to(teacher.device)
    with torch.no_grad():
        teacher_output = teacher(token_ids, output_hidden_states=False, output_attentions=True, use_cache=False)
        teacher_attn_v = teacher_output.attentions[0]  # in the modified model, it will only return one attention, which is the target layer
        t_attn = teacher_attn_v[0]
        t_v = teacher_attn_v[1]

        logging.debug(f"get_teacher_output: any nan: t_attn: {torch.isnan(t_attn).any()}, t_v: {torch.isnan(t_v).any()}")

        t_logits = teacher_output.logits

    return t_logits, t_attn, t_v

def get_student_output(student, token_ids):
    '''
    student: a pretrained model
    token_ids: a tensor of token ids
    student_target_layer: the layer of the teacher model to use as the target

    returns: (logits, q, k, v)
        if used multi-query, q and k will ensure to have same shape
    '''
    device = None
    if hasattr(student, "get_input_device"):
        device = student.get_input_device()
    else: # it means the model is wrapped by DDP
        device = student.module.get_input_device()

    token_ids = token_ids.to(device)
    student_logits = student(token_ids)[0]
    if hasattr(student, "get_intermediate_attn_v"):
        student_attn, student_v = student.get_intermediate_attn_v()
    else: # it means the model is wrapped by DDP
        student_attn, student_v = student.module.get_intermediate_attn_v()
    return student_logits, student_attn, student_v


def loss_soft_logits_kl(student_logits, teacher_logits, temperature=1.0, device=None):
    '''
    student_logits: logits from student model, shape: (batch_size, seq_len, vocab_size)
    teacher_logits: logits from teacher model, shape: (batch_size, seq_len, vocab_size)
    temperature: temperature for soft logits

    returns: loss

    use kl divergence to calculate loss
    '''
    assert student_logits.shape == teacher_logits.shape, f"student_logits.shape: {student_logits.shape}, teacher_logits.shape: {teacher_logits.shape}"

    if device is None:
        device = student_logits.device
    teacher_logits = teacher_logits.to(device)

    vocab_size = student_logits.shape[-1]
    seq_len = student_logits.shape[1]

    #assert student_logits.dtype == teacher_logits.dtype, f"student_logits.dtype: {student_logits.dtype}, teacher_logits.dtype: {teacher_logits.dtype}"

    if temperature != 1.0:
        student_logits = student_logits * (1 / temperature) # shape: (batch_size, seq_len, vocab_size)
        teacher_logits = teacher_logits * (1 / temperature)
    student_prob = torch.nn.functional.log_softmax(student_logits, dim=-1).exp() # equavalent to softmax, slow but more stable than softmax
    teacher_prob = torch.nn.functional.log_softmax(teacher_logits, dim=-1).exp()

    student_prob = student_prob + (1e-10 / vocab_size) # add epsilon to probability (the softmax's result) avoid nan
    teacher_prob = teacher_prob + (1e-10 / vocab_size)

    loss = torch.nn.functional.kl_div(student_prob.log(), teacher_prob, reduction='sum') * (temperature ** 2) / seq_len
    logging.debug(f"loss_soft_logits: {loss.item()}")
    return loss


def loss_soft_logits(student_logits, teacher_logits, temperature=1.0, device=None):
    # useing mse loss
    assert student_logits.shape == teacher_logits.shape, f"student_logits.shape: {student_logits.shape}, teacher_logits.shape: {teacher_logits.shape}"
    loss = torch.nn.functional.mse_loss(student_logits, teacher_logits, reduction='sum') / student_logits.shape[1] / student_logits.shape[2]
    logging.debug(f"loss_soft_logits, MSE: {loss.item()}")
    return loss



def loss_hard_label(student_logits, student_input_tokens, next_token_ids):
    '''
    student_logits: logits from student model
    next_token_ids: next token ids

    returns: loss
    '''
    seq_len = student_logits.shape[1]
    next_token_ids = next_token_ids.to(student_logits.device).long()
    next_token_ids = next_token_ids.unsqueeze(1) # shape: (batch_size, 1)
    target_tokens = student_input_tokens[:, 1:].contiguous() # ignore the first token
    target_tokens = torch.cat([target_tokens, next_token_ids], dim=1) # shape: (batch_size, seq_len)

    target_tokens = target_tokens.view(-1) # shape: (batch_size * seq_len)

    student_logits = student_logits.view(-1, student_logits.shape[-1]) # shape: (batch_size * seq_len, vocab_size)

    ret = torch.nn.functional.cross_entropy(student_logits, target_tokens, reduction='sum') # need to devide batch size
    ret = ret / seq_len
    # ret = ret / 100 # the result of cross_entropy is way larger than the result of kl_div; but their derivative will have same magnitude
    logging.debug(f"loss_hard_label: {ret.item()}")
    return ret

def loss_hard_label_no_last(student_logits, student_input_tokens):
    '''
    student_logits: logits from student model
    next_token_ids: next token ids

    returns: loss
    '''
    seq_len = student_input_tokens.shape[1] - 1
    target_tokens = student_input_tokens[:, 1:].view(-1) # ignore the first token
    student_logits_no_last = student_logits[:, :-1, :].view(-1, student_logits.shape[-1]).contiguous() # shape: (batch_size * seq_len, vocab_size)

    ret = torch.nn.functional.cross_entropy(student_logits_no_last, target_tokens, reduction='sum') # need to devide batch size
    ret = ret / seq_len
    # ret = ret / 100 # the result of cross_entropy is way larger than the result of kl_div; but their derivative will have same magnitude
    return ret


def loss_mimic_attn(student_attn: torch.Tensor, # TODO:  maskout the attention window
                    student_v,
                    teacher_attn,
                    teacher_v,
                    virtual_v_head_num,
                    sliding_window_size,
                    device=None) -> torch.Tensor:
    '''
    student_attn: attention score from student model, shape: (batch_size, n_head, seq_len, seq_len)
    student_v: value from student model, shape: (batch_size, seq_len, student_dim)
    teacher_attn: attention score from teacher model
    teacher_v: value from teacher model, shape: (batch_size, seq_len, teacher_dim)

    NOTE:
    Although this function support multi-query, but a multi-query model may never learn the value vector,
    because it simply repeat the value vector. So should not use multi-query for mimic attention, unless teacher model and
    student model have same behavior when using multi-query.

    Another problem: one model use rotery position embedding, another model use alibi position embedding,
    the only compareable thing is the attention score after softmax.

    q and k must have same shape

    returns: loss
    Need to devide batch size; Not doing it due to mini-batch
    '''
    logging.debug(f"loss_mimic_attn: any nan before start : s_vr: {torch.isnan(student_v).any()}, t_vr: {torch.isnan(teacher_v).any()}, {teacher_v.shape}")
    assert student_attn.shape == teacher_attn.shape, f"student_attn.shape: {student_attn.shape}, teacher_attn.shape: {teacher_attn.shape}"

    if device is None:
        device = student_attn.device

    student_attn = student_attn.to(device)
    student_v = student_v.to(device)
    teacher_attn = teacher_attn.to(device)
    teacher_v = teacher_v.to(device)

    batch_size = student_attn.shape[0]
    q_num = student_attn.shape[2]
    k_num = student_attn.shape[-1]

    n_head = student_attn.shape[1]
    attn_mask = get_attn_mask(k_num, sliding_window_size, device, student_attn.dtype) # shape: (k_num, k_num)
    attn_mask = attn_mask[-q_num: , :]
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) # shape: (1, 1, q_num, k_num)
    attn_mask = attn_mask.expand(batch_size, n_head, q_num, k_num) # shape: (batch_size, n_head, q_num, k_num)
    student_attn = student_attn * attn_mask # apply causal and sliding mask, element-wise multiply
    student_attn = student_attn + (1e-10 / k_num)

    with torch.no_grad():
        teacher_attn = teacher_attn * attn_mask
        teacher_attn = teacher_attn + (1e-10 / k_num) # add epsilon to probability (the softmax's result) avoid nan
    cmp_attn = torch.nn.functional.kl_div(student_attn.log(), teacher_attn, reduction="sum") / n_head / q_num *2 # does not devide batch size


    # compare value vectors, see paper miniLM
    assert student_v.shape[0] == teacher_v.shape[0], f"student_v.shape: {student_v.shape}, teacher_v.shape: {teacher_v.shape}"
    assert student_v.shape[1] == teacher_v.shape[1], f"student_v.shape: {student_v.shape}, teacher_v.shape: {teacher_v.shape}"
    v_seqlen = student_v.shape[1]
    s_v_head_dim = student_v.shape[2] // virtual_v_head_num
    
    s_v = student_v.view(batch_size, v_seqlen, virtual_v_head_num, s_v_head_dim).transpose(1, 2)
    s_vr = torch.matmul(s_v, s_v.transpose(2, 3)) * (1 / math.sqrt(s_v_head_dim)) # (batch_size, virtual_v_head_num, seq_len, seq_len)

    with torch.no_grad():
        t_v_head_dim = teacher_v.shape[2] // virtual_v_head_num
        t_v = teacher_v.view(batch_size, v_seqlen, virtual_v_head_num, t_v_head_dim).transpose(1, 2)
        t_vr = torch.matmul(t_v, t_v.transpose(2, 3)) * (1 / math.sqrt(t_v_head_dim)) # (batch_size, virtual_v_head_num, seq_len, seq_len)


    

    logging.debug(f"loss_mimic_attn: any nan after mul: s_vr: {torch.isnan(s_vr).any()}, t_vr: {torch.isnan(t_vr).any()}")
    assert torch.isnan(s_vr).any() == False, f"loss_mimic_attn: any nan after mul: s_vr: {torch.isnan(s_vr).any()}, t_vr: {torch.isnan(t_vr).any()}"
    assert torch.isnan(t_vr).any() == False, f"loss_mimic_attn: any nan after mul: s_vr: {torch.isnan(s_vr).any()}, t_vr: {torch.isnan(t_vr).any()}"

    # # apply causal mask
    # mask = get_causal_mask(seq_len, device)
    # s_vr = s_vr + mask
    # t_vr = t_vr + mask

    # logging.debug(f"loss_mimic_attn: any nan: mask: {torch.isnan(mask).any()}, s_vr: {torch.isnan(s_vr).any()}, t_vr: {torch.isnan(t_vr).any()}")

    s_vr_score = torch.nn.functional.log_softmax(s_vr, dim=-1).exp()

    # s_vr_score = torch.tril(s_vr_score, diagonal=0) + (1e-8 / seq_len) # add epsilon to probability (the softmax's result) avoid nan
    # t_vr_score = torch.tril(t_vr_score, diagonal=0) + (1e-8 / seq_len)
    s_vr_score = s_vr_score + (1e-10 / v_seqlen)

    with torch.no_grad():
        t_vr_score = torch.nn.functional.log_softmax(t_vr, dim=-1).exp()
        t_vr_score = t_vr_score + (1e-10 / v_seqlen)
    
    logging.debug(f"loss_mimic_attn: any nan after add eps: s_vr: {torch.isnan(s_vr).any()}, t_vr: {torch.isnan(t_vr).any()}")

    cmp_v = torch.nn.functional.kl_div(s_vr_score.log(), t_vr_score, reduction='sum') / virtual_v_head_num / v_seqlen # need to devide batch size
    logging.debug(f"loss_mimic_attn: cmp_attn: {cmp_attn.item()}, cmp_v: {cmp_v.item()}")
    return (cmp_attn + cmp_v)


_attn_mask_cache = {} # the mask that only have 1 and 0. element-wise multiply with attention score to apply causal mask
def get_attn_mask(seq_len, sliding_window_size, device, dtype):
    '''
    return: (seq_len, seq_len) tensor
    '''
    if (seq_len, sliding_window_size) not in _attn_mask_cache:
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=0) # causal mask
        if sliding_window_size < seq_len:
            mask = torch.triu(mask, -sliding_window_size+1) # sliding window mask
        _attn_mask_cache[(seq_len, sliding_window_size)] = mask
    elif _attn_mask_cache[(seq_len, sliding_window_size)].device != device or _attn_mask_cache[(seq_len, sliding_window_size)].dtype != dtype:
        mask = _attn_mask_cache[(seq_len, sliding_window_size)].to(device)
        _attn_mask_cache[(seq_len, sliding_window_size)] = mask
    return _attn_mask_cache[(seq_len, sliding_window_size)]




class Distill_train_config:
    def __init__(self,
                 lr=9e-4, eps=1e-8,
                 beta1=0.9,
                 beta2=0.98, weight_decay=1e-3,
                 opt_name="adamw",
                 soft_loss_weight=0.2,
                 hard_loss_weight=0.1,
                 mimic_loss_weight=0.7,
                 virtual_v_head_num = 16,
                 loss_soft_temperature=1.6,
                 gradient_clip=1.0,
                 llr_warmup_start_factor=0.0001,
                 llr_start_factor=1.0,
                 llr_end_factor=0.001,
                 llr_warmup_iters=5,
                 llr_total_iters=10,
                 llr_last_epoch=-1,
                 slr_seg: list = None,):
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.opt_name = opt_name

        self.soft_loss_weight = soft_loss_weight
        self.hard_loss_weight = hard_loss_weight
        self.mimic_loss_weight = mimic_loss_weight
        self.virtual_v_head_num = virtual_v_head_num
        self.loss_soft_temperature = loss_soft_temperature

        self.gradient_clip = gradient_clip

        self.llr_warmup_start_factor = llr_warmup_start_factor
        self.llr_start_factor = llr_start_factor
        self.llr_end_factor = llr_end_factor
        self.llr_warmup_iters = llr_warmup_iters
        self.llr_total_iters = llr_total_iters
        self.llr_last_epoch = -1 #llr_last_epoch
        self.slr_seg = slr_seg # it will override other learning rate schedule setting, and using the Linear_schedular_seg instead

        self.ckpt_path = None
        self.ckpt_ignore_opt = False
        self.student_dtype = torch.bfloat16
        self.teacher_dtype = torch.bfloat16

    def __str__(self):
        return f"""
        lr: {self.lr}
        eps: {self.eps}
        beta1: {self.beta1}
        beta2: {self.beta2}
        weight_decay: {self.weight_decay}
        soft_loss_weight: {self.soft_loss_weight}
        hard_loss_weight: {self.hard_loss_weight}
        mimic_loss_weight: {self.mimic_loss_weight}
        virtual_v_head_num: {self.virtual_v_head_num}
        loss_soft_temperature: {self.loss_soft_temperature}
        gradient_clip: {self.gradient_clip}
        llr_warmup_start_factor: {self.llr_warmup_start_factor}
        llr_start_factor: {self.llr_start_factor}
        llr_end_factor: {self.llr_end_factor}
        llr_warmup_iters: {self.llr_warmup_iters}
        llr_total_iters: {self.llr_total_iters}
        llr_last_epoch: {self.llr_last_epoch}
        ckpt_path: {self.ckpt_path}
        ckpt_ignore_opt: {self.ckpt_ignore_opt}
        student_dtype: {self.student_dtype}
        teacher_dtype: {self.teacher_dtype}
        """

def _post_process_attn(attn):
    '''
    attn: (batch_size, n_head, seq_len, seq_len)
    '''
    if attn.shape[-2] <= 128: # only return the query attention score of last 64 tokens
        return attn
    return attn[:, :, -128:, :]

    # ret = torch.mean(attn, dim=1, keepdim=True) # it only force student to mimic the average attention score of teacher
    # return ret

def _post_process_v(v):
    '''
    v: (batch_size, seq_len, dim)
    '''
    if v.shape[-2] <= 128:
        return v
    return v[:, -128:, :]

def _post_process_logits(logits):
    '''
    logits: (batch_size, seq_len, vocab_size)
    '''
    # # only keep first 64 and last 64 tokens # TODO: MORE tokens
    # first_part_cnt = 64
    # last_part_cnt = 64
    # if logits.shape[1] <= first_part_cnt + last_part_cnt:
    #     return logits
    # ret = logits[:, 0:first_part_cnt, :]
    # ret = torch.cat([ret, logits[:, -last_part_cnt:, :]], dim=1)
    # return ret

    # return the last 128 logits
    if logits.shape[1] <= 128:
        return logits
    return logits[:, -128:, :]

post_process_attn = _post_process_attn
post_process_v = _post_process_v
post_process_logits = _post_process_logits