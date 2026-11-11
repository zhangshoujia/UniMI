"""Training orchestration and adapter checkpoints for both alignment stages."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, set_seed

from .checkpoint import (
    CONFIG_NAME,
    REPRESENTATION_FIELDS,
    SCHEMA_VERSION,
    adapter_weight_path,
    choose_processor_source,
    load_checkpoint_metadata,
    read_json,
    resolve_model_path,
    resolve_resume_checkpoint,
    sha256,
    validate_adapter,
    validate_resume,
)
from .config import validate_config


class AlignmentTrainer(Trainer):
    """Save the nested PEFT adapter and aggregate count-weighted diagnostics."""

    def __init__(self, *args, checkpoint_metadata: dict, resume_checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_metadata = checkpoint_metadata
        self.preloaded_resume_checkpoint = resume_checkpoint
        self.metric_totals: dict[str, torch.Tensor] = {}
        self.metric_counts: dict[str, torch.Tensor] = {}
        # Our models return a mean over anchors, so Trainer owns accumulation
        # scaling rather than passing a token count into the objective.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]
        if model.training:
            for name, (value, count) in outputs["metrics"].items():
                value = value.detach().float()
                count = torch.as_tensor(count, device=value.device, dtype=torch.float32)
                total = value * count
                if name in self.metric_totals:
                    self.metric_totals[name].add_(total)
                    self.metric_counts[name].add_(count)
                else:
                    self.metric_totals[name] = total.clone()
                    self.metric_counts[name] = count.clone()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach().mean(), None, None

    def log(self, logs, start_time=None):
        if "loss" in logs and self.metric_totals:
            names = sorted(self.metric_totals)
            values = torch.stack(
                [self.metric_totals[name] for name in names]
                + [self.metric_counts[name] for name in names]
            )
            values = self.accelerator.reduce(values, reduction="sum")
            logs = dict(logs)
            for index, name in enumerate(names):
                count = values[len(names) + index]
                if count.item() > 0:
                    logs[name] = (values[index] / count).item()
            self.metric_totals.clear()
            self.metric_counts.clear()
        return super().log(logs, start_time=start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        if not self.args.should_save:
            return
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        model.encoder.model.save_pretrained(str(output))
        model.encoder.processor.save_pretrained(str(output))
        (output / CONFIG_NAME).write_text(
            json.dumps(self.checkpoint_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.args, output / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Trainer sees the outer loss wrapper; the nested adapter was loaded
        # before Trainer construction. It still restores optimizer and RNG.
        checkpoint = Path(resume_from_checkpoint).resolve()
        if checkpoint != self.preloaded_resume_checkpoint:
            raise RuntimeError("The requested resume adapter was not preloaded")
        if model is not None and model is not self.model:
            raise RuntimeError("Unexpected model wrapper during checkpoint restoration")


def _configure_device(config: dict) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if config["bf16"] and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 training requires a supported CUDA GPU; pass --no-bf16 for FP32")
    config["tf32"] = bool(
        config["tf32"] and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )


def _initialize_adapter(encoder, config: dict, source: Path | None) -> None:
    from peft import LoraConfig, PeftModel, get_peft_model

    encoder.model.enable_input_require_grads()
    if source is not None:
        validate_adapter(source)
        encoder.model = PeftModel.from_pretrained(encoder.model, str(source), is_trainable=True)
    else:
        lora = read_json(config["lora_config"])
        for name in ("target_modules", "modules_to_save"):
            modules = lora.get(name) or []
            modules = [modules] if isinstance(modules, str) else modules
            if any("lm_head" in str(module).lower() for module in modules):
                raise ValueError("The item encoder does not train lm_head adapters")
        encoder.model = get_peft_model(encoder.model, LoraConfig(**lora))
    trainable = [
        (name, parameter)
        for name, parameter in encoder.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lm_head" in name.lower() for name, _ in trainable):
        raise ValueError("Expected trainable item-encoder LoRA parameters, excluding lm_head")
    encoder.model.config.use_cache = False


def run_training(config: dict) -> None:
    from .data.cca import CCACollator, CCADataset
    from .data.common import split_dataset
    from .data.spa import SPACollator, SPADataset
    from .encoder import Qwen2VLItemEncoder
    from .models import CCAModel, SPAModel
    from .tokenization import ItemTokenizer

    config = dict(config)
    stage = config["stage"]
    _configure_device(config)
    set_seed(config["seed"])
    output = Path(config["output_dir"])
    resume = resolve_resume_checkpoint(config["resume_from_checkpoint"], output)
    if (
        output.exists()
        and any(output.iterdir())
        and resume is None
        and not config["overwrite_output_dir"]
    ):
        raise FileExistsError(
            f"Output directory is nonempty: {output}; choose a new directory or resume"
        )
    source = None
    if stage == "cca":
        source = Path(config["stage1_checkpoint"])
        if source.resolve() == output.resolve():
            raise ValueError("CCA output directory must differ from the SPA checkpoint")
        stage1 = load_checkpoint_metadata(source)
        if stage1["stage"] != "spa":
            raise ValueError("CCA initialization requires a SPA checkpoint")
        for name in REPRESENTATION_FIELDS:
            if config[name] is not None and config[name] != stage1[name]:
                raise ValueError(f"CCA {name} must match the SPA checkpoint")
            config[name] = stage1[name]
        config["model_path"] = str(resolve_model_path(stage1, config["model_path"]))
        config["stage1_adapter_sha256"] = sha256(adapter_weight_path(source))
    else:
        if not Path(config["model_path"]).is_dir():
            raise FileNotFoundError(f"Base-model directory not found: {config['model_path']}")
        config["lora_config_sha256"] = sha256(config["lora_config"])
    validate_config(config)
    config.update(
        schema_version=SCHEMA_VERSION,
        train_file_sha256=sha256(config["train_file"]),
        base_model_config_sha256=sha256(Path(config["model_path"]) / "config.json"),
    )
    dataset_kwargs = {
        name: config[name]
        for name in ("train_file", "image_root", "max_char_length", "max_train_samples")
    }
    dataset = (
        SPADataset(**dataset_kwargs)
        if stage == "spa"
        else CCADataset(
            **dataset_kwargs, num_negatives=config["num_negatives"], seed=config["seed"]
        )
    )
    train_data, eval_data = split_dataset(dataset, config["validation_ratio"], config["seed"])
    processor_source = Path(config["model_path"])
    if resume is not None:
        processor_source = choose_processor_source(
            resume, load_checkpoint_metadata(resume), processor_source
        )
    elif source is not None:
        processor_source = choose_processor_source(source, stage1, processor_source)
    encoder = Qwen2VLItemEncoder(
        model_path=config["model_path"],
        processor_path=str(processor_source),
        image_min_pixels=config["image_min_pixels"],
        image_max_pixels=config["image_max_pixels"],
        attn_implementation=config["attn_implementation"],
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
        pooling=config["pooling"],
    )
    _initialize_adapter(encoder, config, resume or source)
    config["adapter_base_model"] = encoder.model.peft_config["default"].base_model_name_or_path
    model = (
        SPAModel(encoder, beta=config["beta"])
        if stage == "spa"
        else CCAModel(encoder, temperature=config["temperature"])
    )
    tokenizer = ItemTokenizer(
        encoder.processor,
        config["item_prompt"],
        max_length=config["max_length"],
        max_char_length=config["max_char_length"],
        item_text_mode=config["item_text_mode"],
    )
    collator = SPACollator(tokenizer) if stage == "spa" else CCACollator(tokenizer)
    args = TrainingArguments(
        output_dir=str(output),
        overwrite_output_dir=config["overwrite_output_dir"],
        do_train=True,
        do_eval=eval_data is not None,
        eval_strategy="steps" if eval_data is not None else "no",
        eval_steps=config["eval_steps"],
        prediction_loss_only=True,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        num_train_epochs=config["num_epochs"],
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config["optim"],
        max_grad_norm=config["max_grad_norm"],
        logging_strategy="steps",
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        data_seed=config["seed"],
        bf16=config["bf16"],
        tf32=config["tf32"],
        gradient_checkpointing=False,
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=config["dataloader_num_workers"] > 0,
        remove_unused_columns=False,
        label_names=[],
        report_to=[],
        ddp_find_unused_parameters=False,
    )
    config.update(
        world_size=args.world_size,
        effective_global_batch_size=config["batch_size"]
        * config["gradient_accumulation_steps"]
        * args.world_size,
        train_examples=len(train_data),
        validation_examples=len(eval_data) if eval_data is not None else 0,
    )
    validate_resume(resume, config)
    trainer = AlignmentTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        processing_class=encoder.processor,
        checkpoint_metadata=config,
        resume_checkpoint=resume,
    )
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{stage.upper()}: train={len(train_data)}, validation={config['validation_examples']}, "
            f"global_batch={config['effective_global_batch_size']}, output={output}",
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()
    result = trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
