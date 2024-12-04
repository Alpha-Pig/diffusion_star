import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig, EarlyStoppingCallback
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
import os
import time
import wandb

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--do_eval", type=bool, default=False)
    return parser.parse_args()



args = get_args()

debug = args.debug
do_eval = args.do_eval






random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# MAIN SETUP
root_prefix = "/data_train/yeqigao/code/llms_factory/"
wandb_cache_dir = root_prefix + "cache/quietstar/wandb_cache"
dataset_name = 'open-web-math/open-web-math'

# dataset_name = 'c4'
project_name = "star-2"
os.environ["WANDB_PROJECT"] = project_name + "-" + dataset_name.split("/")[-1]
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
n_ahead_talk_global = 4
n_passes_global = 2
n_ahead_global = 12
n_examples = 1_000
full_batch_size = 4
eval_and_logging_steps = 10
save_steps = 100


def model_init(params):
    original = False
    if params is None:
        params = {}
    else:
        params = params.params
    # save params to file
    n_ahead = params.get("n_ahead", n_ahead_global if not original else 1)
    n_passes = params.get("n_passes", n_passes_global if not original else 1)
    gumbel_temperature = params.get("gumbel_temperature", 1)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    gradient_accumulation_steps = params.get("gradient_accumulation_steps", global_gradient_accumulation_steps)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)

    model_name = "mistralai/Mistral-7B-v0.1"
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map='auto',
        cache_dir=root_prefix + "cache",
    )
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    
    model.n_ahead = n_ahead
    
    model.n_passes = n_passes
    model.n_tokens_print = gradient_accumulation_steps
    model.gradient_accumulation_steps = gradient_accumulation_steps
    model.residual_think_head = residual_think_head
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.gumbel_temperature = gumbel_temperature
    model.wandb_enabled = True
    model.original_mode = original
    model.config_params = params
    model.run_start = int(time.time())
    model.kill_after = 100
    model.train()
    return model

# Load dataset
import time
load_dataset_start = time.time()

if debug:
    dataset = load_dataset(
        path = "/data_train/yeqigao/code/llms_factory/cache/datasets/open-web-math-debug",
        "en" if "c4" in dataset_name else "default",
        split=f"train[:{n_examples}]",
        # ignore_verifications=True,
        num_proc=16,
        cache_dir=root_prefix + "cache/datasets/",
    )

else:
    dataset = load_dataset(
        dataset_name,
        "en" if "c4" in dataset_name else "default",
        split=f"train[:{n_examples}]",
        # ignore_verifications=True,
        num_proc=16,
        cache_dir=root_prefix + "cache/datasets/",
    )
data_process_start = time.time()
train_dataset = dataset.shuffle(seed=random_seed).map(preprocess_function, batched=True, writer_batch_size=200)
load_dataset_finished = time.time()
print(f"Time to load dataset: {load_dataset_finished - load_dataset_start}")
print(f"Time to process dataset: {load_dataset_finished - data_process_start}")

if do_eval:
    from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
    
    eval_dataset_gsm = load_dataset("gsm8k", "main", split="test").map(preprocess_eval_function_gsm, batched=True, writer_batch_size=200)
    eval_dataset_csqa = load_dataset("tau/commonsense_qa", "default", split="validation").map(preprocess_eval_function_csqa, batched=True, writer_batch_size=200)
    
    eval_datasets = {
    "gsm8k": eval_dataset_gsm.select(range(10)),
    "csqa": eval_dataset_csqa.select(range(10)),
}


batch_size = full_batch_size // n_passes_global
global_gradient_accumulation_steps = full_batch_size // batch_size


if __name__ == "__main__":
    import datetime
    model = model_init(None)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    training_args = TrainingArguments(
        output_dir=root_prefix + f"cache/{project_name}/{run_id}",
        learning_rate=1e-6,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=global_gradient_accumulation_steps,
        max_grad_norm=1.0,
        max_steps=100000,
        warmup_steps=20,
        # auto_find_batch_size=True,
        weight_decay=0.001,
        label_names=["labels"],
        include_inputs_for_metrics=True,
        logging_steps=eval_and_logging_steps ,
        eval_steps=eval_and_logging_steps if do_eval else 0,
        evaluation_strategy="steps" if do_eval else "no",
        save_steps=save_steps,
        run_name=f"n={n_ahead_global}_nt={n_ahead_talk_global}_np={n_passes_global}"
    )


    trainer = Trainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets if do_eval else None,
        compute_metrics=compute_metrics,
        model = model
    )

    trainer.train()
