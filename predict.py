import torch
import re
import time
import math
import torch.nn.functional as F

def preprocess_texts(banners, tokenizer, device, max_length=128, verbose=False):
    """批量文本预处理，返回 tensors + offset_mapping(np.array)"""
    if verbose:
        print(f"[preprocess] 编码 {len(banners)} 条文本，max_length={max_length}")
        t0 = time.time()
    batch_encodings = tokenizer(
        banners,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    if verbose:
        print(f"[preprocess] 完成，耗时 {time.time()-t0:.2f}s")
    return {
        "input_ids": batch_encodings.input_ids.to(device),
        "attention_mask": batch_encodings.attention_mask.to(device),
        "offset_mapping": batch_encodings.offset_mapping.cpu().numpy()
    }

def extract_entities(tokens, tags, offset_mapping, original_text, token_confidences=None):
    """从 token/tag/offset 提取实体字符串及置信度，返回 list of (str, float)"""
    device_chunks = []
    current_chunk = []
    for idx, (token, tag, off) in enumerate(zip(tokens, tags, offset_mapping)):
        start, end = int(off[0]), int(off[1])
        if tag == 'B-DEVICE':
            if current_chunk:
                device_chunks.append(current_chunk)
            current_chunk = [(idx, token, start, end)]
        elif tag == 'I-DEVICE' and current_chunk:
            current_chunk.append((idx, token, start, end))
        elif tag == 'E-DEVICE' and current_chunk:
            current_chunk.append((idx, token, start, end))
            device_chunks.append(current_chunk)
            current_chunk = []
        else:
            if current_chunk:
                device_chunks.append(current_chunk)
                current_chunk = []
    if current_chunk:
        device_chunks.append(current_chunk)

    device_mentions = []
    for chunk in device_chunks:
        starts = [c[2] for c in chunk]
        ends = [c[3] for c in chunk]
        start_pos = min(starts)
        end_pos = max(ends)
        if end_pos <= start_pos:
            continue
        device_str = original_text[start_pos:end_pos].strip()
        if not device_str:
            continue
        conf = None
        if token_confidences is not None:
            idxs = [c[0] for c in chunk]
            vals = [float(token_confidences[i]) for i in idxs if i < len(token_confidences)]
            if vals:
                conf = float(sum(vals) / len(vals))
        device_mentions.append((device_str, conf if conf is not None else 0.0))
    return device_mentions

def compute_crf_marginals(emissions, mask, transitions):
    """
    emissions: (B, L, K)
    mask: (B, L) bool
    transitions: (K, K)
    返回 marginals (B, L, K), log_Z (B,)
    """
    B, L, K = emissions.shape
    device = emissions.device
    emissions_t = emissions.transpose(0, 1)  # (L, B, K)
    mask_t = mask.transpose(0, 1)            # (L, B)

    alpha = emissions_t[0]  # (B, K)
    for t in range(1, L):
        scores = alpha.unsqueeze(2) + transitions.unsqueeze(0)  # (B, K, K)
        scores = torch.logsumexp(scores, dim=1) + emissions_t[t]  # (B, K)
        mask_cur = mask_t[t].unsqueeze(1)
        alpha = torch.where(mask_cur, scores, alpha)
    log_Z = torch.logsumexp(alpha, dim=1)  # (B,)

    beta = torch.zeros(B, K, device=device)
    betas = [beta]
    for t in range(L-1, 0, -1):
        scores = transitions.unsqueeze(0) + emissions_t[t].unsqueeze(0) + beta.unsqueeze(1)  # (B, K_prev, K_curr)
        scores = torch.logsumexp(scores, dim=2)
        mask_cur = mask_t[t].unsqueeze(1)
        beta = torch.where(mask_cur, scores, beta)
        betas.append(beta)
    betas = list(reversed(betas))

    alphas = []
    alpha = emissions_t[0]
    alphas.append(alpha)
    for t in range(1, L):
        scores = alpha.unsqueeze(2) + transitions.unsqueeze(0)
        alpha = torch.logsumexp(scores, dim=1) + emissions_t[t]
        mask_cur = mask_t[t].unsqueeze(1)
        alpha = torch.where(mask_cur, alpha, alphas[-1])
        alphas.append(alpha)
    alphas = torch.stack(alphas, dim=1)  # (B, L, K)
    betas = torch.stack(betas, dim=1)    # (B, L, K)

    marginals_log = alphas + betas - log_Z.unsqueeze(1).unsqueeze(2)
    marginals = torch.exp(marginals_log) * mask.transpose(0,1).unsqueeze(2).to(device)
    return marginals, log_Z

def sync_sliding_inference(model, tokenizer, seqlen, stride, text):
    """
    单条文本滑动窗口推理，返回 {'bio_tags': [...], 'extracted_models': [(str,conf), ...]}
    bio_tags 为 token 级别字符串标签（与 offset_mapping 对齐）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    encoding = tokenizer(text, truncation=False, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]
    total_len = len(input_ids)
    if total_len == 0:
        return {"bio_tags": [], "extracted_models": []}

    bio_tags_int = [0] * total_len
    confidences = [0.0] * total_len

    for start in range(0, total_len, stride):
        end = min(start + seqlen, total_len)
        window_ids = input_ids[start:end]
        window_offsets = offset_mapping[start:end]
        window_input_ids = [tokenizer.cls_token_id] + window_ids + [tokenizer.sep_token_id]
        window_attention = [1] * len(window_input_ids)
        padding = seqlen - len(window_input_ids)
        if padding > 0:
            window_input_ids += [tokenizer.pad_token_id] * padding
            window_attention += [0] * padding
        else:
            window_input_ids = window_input_ids[:seqlen]
            window_attention = window_attention[:seqlen]

        input_tensor = torch.tensor([window_input_ids], device=device)
        att_tensor = torch.tensor([window_attention], device=device)

        with torch.no_grad():
            emissions = model(input_ids=input_tensor, attention_mask=att_tensor)  # (1, L, K)
            preds = model.crf.decode(emissions, mask=att_tensor.bool())[0]  # length L
            transitions = model.crf.transitions.data
            marginals_win, _ = compute_crf_marginals(emissions, att_tensor.bool().to(emissions.device), transitions)
            marginals_win = marginals_win.cpu().numpy()[0]  # (L, K)

        # preds index 0 is CLS; align tokens starting i=1 -> global start
        usable = min(len(window_ids), len(preds)-2)  # exclude CLS and SEP
        for i in range(usable):
            pred_label = int(preds[i+1])
            gidx = start + i
            conf = 0.0
            try:
                conf = float(marginals_win[i+1, pred_label])
            except Exception:
                conf = 0.0
            if conf > confidences[gidx]:
                confidences[gidx] = conf
                bio_tags_int[gidx] = pred_label

    # 转换为字符串标签并过滤无效 token（offset 全为0视为无效）
    label_map = {0: "O", 1: "B-DEVICE", 2: "I-DEVICE", 3: "E-DEVICE"}
    bio_tags_str = [label_map.get(t, "O") for t in bio_tags_int]

    filtered_tokens = []
    filtered_tags = []
    filtered_offsets = []
    filtered_confs = []
    for tok_id, tag_str, off, conf in zip(input_ids, bio_tags_str, offset_mapping, confidences):
        if off[0] == 0 and off[1] == 0:
            continue
        filtered_tokens.append(tokenizer.convert_ids_to_tokens([tok_id])[0])
        filtered_tags.append(tag_str)
        filtered_offsets.append(off)
        filtered_confs.append(conf)

    extracted = extract_entities(
        tokens=filtered_tokens,
        tags=filtered_tags,
        offset_mapping=filtered_offsets,
        original_text=text,
        token_confidences=filtered_confs
    )

    dedup = {}
    for s, c in extracted:
        if s in dedup:
            dedup[s] = max(dedup[s], c)
        else:
            dedup[s] = c
    extracted_models = [(k, v) for k, v in dedup.items()]

    return {"bio_tags": bio_tags_str, "extracted_models": extracted_models}

def predict_devices_with_sliding_window(model, tokenizer, banners, seqlen=128, stride=64, verbose=True):
    """
    统一接口：对 texts（list[str]）做预测，始终返回 list of dict:
      {'bio_tags': [...], 'extracted_models': [(str, conf), ...]}
    优化：当所有文本长度 <= seqlen 时使用批量一次性编码推理（更快），否则逐条滑窗。
    与 run.py / com_pred.py 的调用签名一致。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    results = []
    t0 = time.time()

    # 先判断长度，避免对每条重复 tokenizer 编码代价
    lengths = []
    for t in banners:
        try:
            lengths.append(len(tokenizer(t, add_special_tokens=False)["input_ids"]))
        except Exception:
            lengths.append(0)

    if lengths and max(lengths) <= seqlen:
        # 批量一次性编码并批量推理
        if verbose:
            print("[predict] 所有文本可一次编码，使用批量推理")
        inputs = preprocess_texts(banners, tokenizer, device, max_length=seqlen, verbose=verbose)
        with torch.no_grad():
            emissions = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])  # (B, L, K)
            preds_batch = model.crf.decode(emissions, mask=inputs["attention_mask"].bool())  # list of lists
            transitions = model.crf.transitions.data
            marginals, _ = compute_crf_marginals(emissions, inputs["attention_mask"].bool().to(emissions.device), transitions)

        B = len(banners)
        for i in range(B):
            seq_len = int(inputs["attention_mask"][i].sum().item())
            preds = preds_batch[i][:seq_len]
            pred_tags = [ {0:"O",1:"B-DEVICE",2:"I-DEVICE",3:"E-DEVICE"}.get(t_, "O") for t_ in preds ]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i][:seq_len])
            token_conf_all = [float(marginals[i, t_idx, preds[t_idx]].cpu().item()) if t_idx < len(preds) else 0.0 for t_idx in range(seq_len)]

            filtered_tokens = []
            filtered_tags = []
            filtered_offsets = []
            filtered_confidences = []
            for t_idx, (token, tag, offset) in enumerate(zip(tokens, pred_tags, inputs["offset_mapping"][i][:seq_len])):
                if offset[0] == 0 and offset[1] == 0:
                    continue
                filtered_tokens.append(token)
                filtered_tags.append(tag)
                filtered_offsets.append(offset)
                filtered_confidences.append(token_conf_all[t_idx])

            device_mentions = extract_entities(
                tokens=filtered_tokens,
                tags=filtered_tags,
                offset_mapping=filtered_offsets,
                original_text=banners[i],
                token_confidences=filtered_confidences
            )

            dedup = {}
            for s, c in device_mentions:
                if s in dedup:
                    dedup[s] = max(dedup[s], c)
                else:
                    dedup[s] = c
            results.append({"bio_tags": pred_tags, "extracted_models": [(k, v) for k, v in dedup.items()]})
    else:
        # 部分文本过长，逐条滑动窗口推理
        if verbose:
            print("[predict] 存在过长文本，使用滑动窗口逐条推理")
        for idx, text in enumerate(banners):
            if verbose and (idx % 50 == 0):
                elapsed = time.time() - t0
                print(f"[sliding_predict] {idx}/{len(banners)} 处理，已用 {elapsed:.1f}s")
            results.append(sync_sliding_inference(model, tokenizer, seqlen, stride, text))

    if verbose:
        print(f"[predict] 完成 {len(banners)} 条，耗时 {time.time()-t0:.1f}s")
    return results

# end of file
