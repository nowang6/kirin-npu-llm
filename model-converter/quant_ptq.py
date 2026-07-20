
import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from config import DEFAULT_MODEL_PATH, DEFAULT_DOPT_CONFIG, DEFAULT_OPTIMIZE_CONFIG, DEFAULT_OUTPUT_DIR, DEFAULT_GROUP_SIZE, DEFAULT_BLOCK_SIZE, DEFAULT_ACT_BITS, DEFAULT_W_BITS
import torch
import yaml
from dopt.dopt_lm.do_opt import (
    generate_config_file,
    generate_quant_params,
    optimize_model,
    set_calibrate_state,
    set_quant_state,
)
from dopt.dopt_lm.train import get_processed_dataset
from dopt.log import Logger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Plugin-style LLM PTQ")
    parser.add_argument("--model-path", type=str, default="./Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dopt-config", type=str, default="dopt_conf.json")
    parser.add_argument("--optimize-config", type=str, default="optimize-config.yaml")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--act-bits", type=int, default=16)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="dopt_out")
    parser.add_argument(
        "--quant-stage",
        type=str,
        default="stage1",
        choices=["gen_config", "stage1", "stage2", "stage3"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu / auto (default: cuda:0 if available else cpu)",
    )
    return parser.parse_args()


@dataclass
class DeviceLayout:
    primary: str


def load_optimize_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_custom_env(group_size, block_size, act_bits, w_bits):
    os.environ["custom_group_size"] = str(group_size)
    os.environ["custom_block_size"] = str(block_size)
    os.environ["custom_act_bits"] = str(act_bits)
    os.environ["custom_w_bits"] = str(w_bits)


def resolve_device(device_arg):
    if device_arg:
        if device_arg in ("cuda", "gpu"):
            return "cuda:0"
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_layout(device_arg):
    return DeviceLayout(resolve_device(device_arg))


def load_float_model(model_path, layout):
    if layout.primary == "cpu":
        dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        model = model.to("cpu")
    else:
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
        model = model.to(layout.primary)
    model.eval()
    return model


def model_input_device(model):
    return next(model.parameters()).device


def get_quant_model(model, dopt_config):
    if not os.path.exists(dopt_config):
        generate_config_file(model, dopt_config)
        Logger.info(
            "Generated plugin quant config at %s; set quant_strategy then re-run.",
            dopt_config,
        )
        sys.exit(0)
    return optimize_model(model, dopt_config)


def build_dataset(tokenizer, optimize_cfg, max_samples, cutoff_len):
    dataset_config = {"cutoff_len": cutoff_len}
    if optimize_cfg.get("num_samples") is not None:
        dataset_config["num_samples"] = optimize_cfg["num_samples"]
    train_files = optimize_cfg["dataset"]["train_files"]
    return get_processed_dataset(
        train_files,
        tokenizer,
        dataset_config,
        train_samples=max_samples,
    )


def collate_batch(features):
    input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_calibration(model, dataset, input_device, batch_size=1):
    if len(dataset) == 0:
        raise RuntimeError("Calibration dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = {k: v.to(input_device) for k, v in batch.items()}
            model(**batch)
            if (step + 1) % 50 == 0:
                Logger.info("Calibration progress: %d / %d", step + 1, len(loader))


def save_state_dict(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    Logger.info("Saved checkpoint: %s", path)


def load_state_dict(model, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    Logger.info("Loaded checkpoint: %s", path)


def stage1_weight_quant(args, optimize_cfg):
    layout = resolve_layout(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_float_model(args.model_path, layout)
    model = get_quant_model(model, args.dopt_config)

    set_quant_state(model, weight_state=True, input_state=False)
    set_calibrate_state(model, True)

    max_samples = optimize_cfg["dataset"].get("ptq_samples", 1024)
    cutoff_len = optimize_cfg.get("cutoff_len", 128)
    dataset = build_dataset(tokenizer, optimize_cfg, max_samples, cutoff_len)

    Logger.info("Stage1 (PTQ): calibrating weights on %d samples", len(dataset))
    run_calibration(model, dataset, model_input_device(model))

    set_calibrate_state(model, False)
    save_state_dict(model, os.path.join(args.output_dir, "trained_quant_weight.pth"))


def stage2_activation_quant(args, optimize_cfg):
    layout = resolve_layout(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_float_model(args.model_path, layout)
    model = optimize_model(model, args.dopt_config)

    weight_ckpt = os.path.join(args.output_dir, "trained_quant_weight.pth")
    if not os.path.exists(weight_ckpt):
        raise FileNotFoundError(f"Missing {weight_ckpt}; run stage1 first.")

    load_state_dict(model, weight_ckpt)
    set_quant_state(model, weight_state=True, input_state=True)
    set_calibrate_state(model, True)

    max_samples = optimize_cfg.get("num_samples", 256)
    cutoff_len = optimize_cfg.get("cutoff_len", 128)
    dataset = build_dataset(tokenizer, optimize_cfg, max_samples, cutoff_len)

    Logger.info("Stage2 (PTQ): calibrating activations on %d samples", len(dataset))
    run_calibration(model, dataset, model_input_device(model))

    set_calibrate_state(model, False)
    save_state_dict(model, os.path.join(args.output_dir, "trained.pth"))


def stage3_export_params(args, optimize_cfg):
    layout = resolve_layout(args.device)
    model = load_float_model(args.model_path, layout)
    model = optimize_model(model, args.dopt_config)

    trained_ckpt = os.path.join(args.output_dir, "trained.pth")
    if not os.path.exists(trained_ckpt):
        raise FileNotFoundError(f"Missing {trained_ckpt}; run stage2 first.")

    load_state_dict(model, trained_ckpt)
    set_quant_state(model, weight_state=True, input_state=True)
    set_calibrate_state(model, False)

    quant_param_2 = optimize_cfg.get("quant_param_2", False)
    embedding_separate = optimize_cfg.get("embedding_separate", True)
    lm_head = optimize_cfg.get("lm_head_size")
    Logger.info(
        "Stage3 (PTQ): generate_quant_params (quant_param_2=%s, embedding_separate=%s)",
        quant_param_2,
        embedding_separate,
    )
    generate_quant_params(
        model,
        args.output_dir,
        quant_param_2=quant_param_2,
        embedding_separate=embedding_separate,
        lm_head=lm_head,
    )


def gen_config_only(args):
    layout = resolve_layout(args.device)
    model = load_float_model(args.model_path, layout)
    os.makedirs(os.path.dirname(args.dopt_config) or ".", exist_ok=True)
    if os.path.exists(args.dopt_config):
        Logger.info("Config already exists: %s", args.dopt_config)
        return
    generate_config_file(model, args.dopt_config)
    Logger.info("Generated config template: %s", args.dopt_config)


def main():
    args = parse_args()
    apply_custom_env(args.group_size, args.block_size, args.act_bits, args.w_bits)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.quant_stage == "gen_config":
        gen_config_only(args)
        return

    if not os.path.exists(args.optimize_config):
        raise FileNotFoundError(args.optimize_config)
    optimize_cfg = load_optimize_config(args.optimize_config)
    config_snapshot = os.path.join(args.output_dir, "config.yaml")
    if not os.path.exists(config_snapshot):
        shutil.copy(args.optimize_config, config_snapshot)

    if not os.path.exists(args.dopt_config):
        layout = resolve_layout(args.device)
        model = load_float_model(args.model_path, layout)
        get_quant_model(model, args.dopt_config)

    if args.quant_stage == "stage1":
        stage1_weight_quant(args, optimize_cfg)
    elif args.quant_stage == "stage2":
        stage2_activation_quant(args, optimize_cfg)
    elif args.quant_stage == "stage3":
        stage3_export_params(args, optimize_cfg)
    else:
        raise ValueError(f"Unknown quant stage: {args.quant_stage}")


if __name__ == "__main__":
    main()
