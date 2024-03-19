# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config


class token():
    def __init__(self, tokenizer):
        self.sp_model = tokenizer
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.unk_id()
        print(self.eos_id)
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        role_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"] + role_special_tokens
        self.special_tokens = {}
        self.index_special_tokens = {}
        for token in special_tokens:
            self.special_tokens[token] = self.n_words
            self.index_special_tokens[self.n_words] = token
            self.n_words += 1

    def encode(self, sample, device):
        if "user" in sample:
            prompt_tokens = [self.special_tokens["<|system|>"]] + self.sp_model.encode("\n") + \
                            self.sp_model.encode(sample["prompt"])
            input_tokens = [self.special_tokens["<|user|>"]] + self.sp_model.encode("\n") + \
                           self.sp_model.encode(sample["user"])

            src_tokens = prompt_tokens + input_tokens
        else:
            prompt_tokens = [self.special_tokens["<|user|>"]] + self.sp_model.encode("\n") + \
                            self.sp_model.encode(sample["prompt"])
            src_tokens = prompt_tokens
        tgt_tokens = [self.special_tokens["<|assistant|>"]] + self.sp_model.encode("\n")
        input_ids = [self.special_tokens["[gMASK]"],
                     self.special_tokens["sop"]] + src_tokens + tgt_tokens
        return torch.tensor(input_ids, dtype=torch.int, device=device)

    def decode(self, t):
        text, buffer = "", []
        for token in t:
            if token in self.index_special_tokens:
                if buffer:
                    text += self.sp_model.decode(buffer)
                    buffer = []
                text += self.index_special_tokens[token]
            else:
                buffer.append(token)
        if buffer:
            text += self.sp_model.decode(buffer)
        return text


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import ChatGLMForConditionalGeneration


def multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model: ChatGLMForConditionalGeneration, x: torch.Tensor, input_pos: torch.Tensor,
            **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model: ChatGLMForConditionalGeneration, x: torch.Tensor, input_pos: torch.Tensor,
                     **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(model: ChatGLMForConditionalGeneration, cur_token: torch.Tensor, input_pos: torch.Tensor,
                    num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False,
                                            enable_math=True):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
            if cur_token == 2:
                return new_tokens, new_probs
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def speculative_decode(
        model: ChatGLMForConditionalGeneration,
        draft_model: ChatGLMForConditionalGeneration,
        cur_token: torch.Tensor,
        input_pos: int,
        speculate_k: int,
        **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)

    t0 = time.perf_counter()  ##
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k,
                                                **sampling_kwargs)

    real_k = len(draft_tokens)
    print(f"Draft decode use time : ${time.perf_counter() - t0}  real_k :{real_k}")

    draft_last_token = draft_tokens[-1]
    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens

    t0 = time.perf_counter()
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + real_k + 1, device=cur_token.device)
    )
    print(f"model forward decode use time : ${time.perf_counter() - t0}")

    t0 = time.perf_counter()
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    print(f"logits_to_probs use time : ${time.perf_counter() - t0}")

    t0 = time.perf_counter()
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, real_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, real_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:real_k] / p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()
    print(f"rejected_locations use time : ${time.perf_counter() - t0}")

    if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
        accept_length = real_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        t0 = time.perf_counter()
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + real_k,
        )
        print(f"model forward decode use time 2 : ${time.perf_counter() - t0}")
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


@torch.no_grad()
def generate(
        model: ChatGLMForConditionalGeneration,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        interactive: bool,
        draft_model: ChatGLMForConditionalGeneration,
        speculate_k: Optional[int] = 8,
        callback=lambda x: x,
        **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens

    T = prompt.size(0)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        print("##################################")
        model.transformer.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.transformer.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    t0 = time.perf_counter()
    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    prefill_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if is_speculative:
        prefill(draft_model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    draft_prefill_time = time.perf_counter() - t0
    seq[T] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())
            eos = 2
            t0 = time.perf_counter()
            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )
            print(f"speculative_decode all use time : ${time.perf_counter() - t0}")
            accept_counts[len(next_tokens) - 1] += 1

            is_eos = False
            if eos is not None and torch.any(next_tokens == eos):
                eos_index = torch.where(next_tokens == eos)[0]
                next_tokens = next_tokens[:eos_index]
                is_eos = True

            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1: input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added, ]:
                callback(i)

            if is_eos:
                seq = seq[: input_pos + num_added + 1]
                break

            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1,
                                              callback=callback, **sampling_kwargs)
        seq[T + 1:T + len(generated_tokens) + 1] = torch.cat(generated_tokens)
        seq = seq[:T + len(generated_tokens) + 1]

    generate_stats = {
        'accept_counts': accept_counts,
        'prefill_time': prefill_time,
        'draft_prefill_time': draft_prefill_time
    }
    return seq, generate_stats


def encode_tokens(tokenizer, string, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp):
    print(checkpoint_path)
    with torch.device('meta'):
        model = ChatGLMForConditionalGeneration.from_name("chatglm3-6B")

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"


def main(
        prompt: str = "Hello, my name is",
        interactive: bool = False,
        num_samples: int = 5,
        max_new_tokens: int = 100,
        top_k: int = 200,
        temperature: float = 0.8,
        checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        compile: bool = True,
        compile_prefill: bool = False,
        profile: Optional[Path] = None,
        draft_checkpoint_path: Optional[Path] = None,
        speculate_k: int = 5,
        device='cuda',
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = "../model/ChatGLM3-6B/tokenizer.model"

    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    print("------", rank)
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenize = token(tokenizer)
    # prompt =  {
    #     "prompt": "你是一位接口设计专家，有丰富的符号语言和规则语法使用经验，善于把需求转化为规范完整的面向编程的接口。\n现在有一个专门标注语言，把自然语言描述的需求转变为json描述的接口定义。该任务的专用标记语言的规则如下\n1）工具和方法描述规则：\ng:namespace.tool_name.function_name\ng是保留关键字，表示goal\n\n2）参数描述规则：\nv:namespace.tool_name.parameter_name \ng:namespace.tool_name.parameter_name \nv和g是关键字，v用于表示基类参数名称，g用于表示派生类的参数名称\n例如：\nv:viv.time.DateTime  #基类时间参数\ng:viv.calendarApp.DateTime #工具calendarApp里的派生时间参数\n\n3）工具调用描述规则：\n[g:namespace.tool_name.function_name] xxx (parameter xxx) (v:namespace.tool_name.parameter_name) xxx {[g:namespace.tool_name.parameter_name] (parameter xxx) [v:namespace.tool_name.parameter_name]} xxx.\n即在task_query文本上增加标记语言，开头中括号内描述完成该任务需要用到的工具和工具内的方法，每个参数首先把文本paramter xxx用小括号括起来，如果是基类参数，在该参数文本后面用小括号标记参数描述；如果是派生类参数，在该参数的前后分别依次标记派生类和基类参数描述，并用花括号把整个参数描述括起来。\n\n现在，你会接收到用户这种格式的接口描述任务\n{\"task_query\": \"xxx\"}\n\n请按如下格式回复json的接口描述\n{\"task_query\": \"xxx\", #用户用语言描述他想用工具完成的任务\n\"target_tool_function\": \"xxx\", #输出应使用的工具和工具里的方法\n\"input_arguments\": \"[xxx]\", #输出该工具需要的参数列表\n\"call_command\": \"[target_tool_function]task_query_with_input_argumets_labelled\" #输出该任务专用的工具调用标记语言\n}\n注意，保留中括号作为符号标记，call_command\"字段直接使用\"target_tool_function\"和\"input_arguments\"字段的内容对\"task_query\"字段的文本进行转换，\"task_query\"字段识别出来的参数文本需要用小括号括起来\n",
    #     "user": "{\n  \"task_query\": \"设置一个下午7点打球的闹钟\"\n}",
    #     "response": "{\n  \"task_query\": \"设置一个下午7点打球的闹钟\",\n  \"target_tool_function\": \"[g:viv.clockApp.SetAlarm]\",\n  \"input_arguments\": [\n    \"(下午7点)[v:viv.clockApp.DetachedTime]\",\n    \"(打球)[v:viv.clockApp.AlarmName]\"\n  ],\n  \"call_command\": \"[g:viv.clockApp.SetAlarm] 设置一个(下午7点)[v:viv.clockApp.DetachedTime](打球)[v:viv.clockApp.AlarmName]的闹钟\"\n}"
    # }

    # encoded = tokenize.encode(prompt, device=device)

    torch.manual_seed(1234)
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    if compile:
        if is_speculative and use_tp:  # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False  # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
        'prefill_time': [],
        'draft_prefill_time': [],
    }

    start = -1 if compile else 0
# 需要更新自己的数据
    prompts = [{
        "prompt": "",
        "response": "",
    },
        {
            "prompt": "",
            "response": "",
        },
        {
            "prompt": "",
            "response": "",
        },
        {
            "prompt": "",
            "response": "",
        },
        {
            "prompt": "",
            "response": "",
        },
        {
            "prompt": "",
            "response": "",
        }]

    inference_time_list = []
    all_inference_time_list = []
    token_time_list = []
    for i in range(num_samples):
        print("-----------------------------------------------------------")
        device_sync(device=device)  # MKG
        print(i)
        print(prompts[i]["prompt"])
        print("*************************")
        prompt = prompts[i]

        start_time = time.perf_counter()
        encoded = tokenize.encode(prompt, device=device)
        prompt_length = encoded.size(0)

        callback = lambda x: x
        token_time_list.append(time.perf_counter() - start_time)
        t0 = time.perf_counter()
        import contextlib
        prof = contextlib.nullcontext()

        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
            aggregate_metrics['prefill_time'].append(metrics['prefill_time'])
            aggregate_metrics['draft_prefill_time'].append(metrics['draft_prefill_time'])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        inference_time_list.append(time.perf_counter() - t0)
        all_inference_time_list.append(time.perf_counter() - start_time)
        t = time.perf_counter() - t0

        if not interactive:
            print(tokenize.decode(y.tolist()))
        else:
            print()
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec \n"
              f"prompt_length:{prompt_length} tokens_generated:{tokens_generated}\n"
              f" prefill_time: {aggregate_metrics['prefill_time'][-1]},  draft_prefill_time: {aggregate_metrics['draft_prefill_time'][-1]}, \n"
              )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)]) / sum(counts_aggregated)}")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    print(
        f"inference_time: {inference_time_list[1:]},  Average inference_time: {torch.mean(torch.tensor(inference_time_list[1:])).item():.2f}, \n")
    print(
        f"All inference_time: {all_inference_time_list[1:]},  Average all inference_time: {torch.mean(torch.tensor(all_inference_time_list[1:])).item():.2f}, \n")
    print(
        f"prefill_time: {aggregate_metrics['prefill_time']},  draft_prefill_time: {aggregate_metrics['draft_prefill_time']}, \n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path(
        "../ChatGLM-Finetuning-master/output/output-glm3_20240307_35cap_multidata_completeutt_32_4096/epoch-20-step-680/chatglm3.pth"))
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true',
                        help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=Path("../model_save/chatglm3_int4.g32.pth"),
                        help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default="cuda", help='device to use')

    args = parser.parse_args()
    args.draft_checkpoint_path = None
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile,
        args.draft_checkpoint_path,
        args.speculate_k, args.device
    )