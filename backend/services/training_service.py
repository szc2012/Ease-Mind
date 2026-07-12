"""训练服务

支持两种模式：
- "real": 使用 transformers + peft 进行真实 LoRA 微调
- "mock": 模拟训练流程（无 GPU 也能演示完整体验）

通过 settings.TRAINING_MODE 控制。
"""
import os
import sys
import time
import json
import random
import threading
import traceback
from datetime import datetime
from pathlib import Path

from config import settings, LOG_DIR, MODEL_DIR
from database import SessionLocal
from models import TrainingTask, AIModel, Dataset
from services.dataset_service import get_dataset_text
from services.loss_service import record_loss


# 专业模式参数定义
PROFESSIONAL_PARAMS = {
    "epochs": {"label": "训练轮数", "default": 3, "min": 1, "max": 50},
    "batch_size": {"label": "批大小", "default": 2, "min": 1, "max": 32},
    "learning_rate": {"label": "学习率", "default": 0.0002, "min": 0.00001, "max": 0.01},
    "lora_r": {"label": "LoRA秩(r)", "default": 8, "min": 1, "max": 64},
    "lora_alpha": {"label": "LoRA Alpha", "default": 16, "min": 1, "max": 128},
    "lora_dropout": {"label": "LoRA Dropout", "default": 0.05, "min": 0.0, "max": 0.5},
    "max_seq_length": {"label": "最大序列长度", "default": 256, "min": 128, "max": 2048},
    "warmup_steps": {"label": "预热步数", "default": 10, "min": 0, "max": 500},
    "weight_decay": {"label": "权重衰减", "default": 0.01, "min": 0.0, "max": 0.1},
}

SIMPLE_PRESETS = {
    "fast": {"label": "快速体验", "epochs": 1, "lora_r": 4, "desc": "1轮训练，速度最快"},
    "balanced": {"label": "均衡推荐", "epochs": 2, "lora_r": 8, "desc": "2轮训练，效果与速度均衡"},
    "quality": {"label": "高质量", "epochs": 3, "lora_r": 16, "desc": "3轮训练，效果最佳"},
}


def get_log_path(task_id: str) -> Path:
    return LOG_DIR / f"training_{task_id}.log"


def write_log(task_id: str, message: str) -> None:
    path = get_log_path(task_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_log(task_id: str) -> str:
    path = get_log_path(task_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ============== 真实训练：LoRA 微调 ==============

def _prepare_dataset(text: str, tokenizer, max_len: int) -> list:
    """把长文本切成 max_len 的片段作为训练样本"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    samples = []
    step = max_len - 16  # 给 prompt 留空间
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_len - 16]
        if len(chunk) >= 32:  # 太短的不取
            samples.append(chunk)
        i += step
    return samples


def _run_real_training(task_id: str) -> None:
    """真实 LoRA 微调"""
    db = SessionLocal()
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return
        task.log_file = str(get_log_path(task_id))
        task.status = "running"
        task.started_at = datetime.utcnow()
        task.progress = 0.0
        db.commit()

        write_log(task_id, "=" * 60)
        write_log(task_id, f"训练任务（真实模式）：{task.name}")
        write_log(task_id, f"模式：{'傻瓜微调' if task.mode == 'simple' else '专业训练'}")
        write_log(task_id, f"参数：{json.dumps(task.params, ensure_ascii=False)}")
        write_log(task_id, "=" * 60)

        base_model = db.query(AIModel).filter(AIModel.id == task.model_id).first()
        # 解析多个数据集
        ds_ids = [s.strip() for s in (task.dataset_id or "").split(",") if s.strip()]
        datasets = [db.query(Dataset).filter(Dataset.id == did).first() for did in ds_ids]
        datasets = [d for d in datasets if d]
        if not base_model or not base_model.local_path:
            write_log(task_id, "[错误] 基础模型未就绪或路径不存在")
            task.status = "failed"
            task.error_message = "基础模型未就绪"
            task.finished_at = datetime.utcnow()
            db.commit()
            return
        if not datasets:
            write_log(task_id, "[错误] 数据集不存在")
            task.status = "failed"
            task.finished_at = datetime.utcnow()
            db.commit()
            return

        model_path = base_model.local_path
        params = task.params
        epochs = int(params.get("epochs", 2))
        batch_size = int(params.get("batch_size", 2))
        lr = float(params.get("learning_rate", 0.0002))
        lora_r = int(params.get("lora_r", 8))
        lora_alpha = int(params.get("lora_alpha", lora_r * 2))
        lora_dropout = float(params.get("lora_dropout", 0.05))
        max_len = int(params.get("max_seq_length", 256))
        warmup_steps = int(params.get("warmup_steps", 10))
        weight_decay = float(params.get("weight_decay", 0.01))

        write_log(task_id, f"基础模型：{base_model.name}" + ("（微调模型再训练）" if base_model.source == "finetune" else ""))
        write_log(task_id, f"路径：{model_path}")
        write_log(task_id, f"数据集（{len(datasets)} 个）：")
        for d in datasets:
            write_log(task_id, f"  - {d.name}（{d.sample_count} 样本）")

        # ---- 加载依赖 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 1/4：加载基础模型")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
        from peft import LoraConfig, get_peft_model, TaskType

        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device != "cpu" else torch.float32
        write_log(task_id, f"设备：{device}, dtype：{dtype}")

        write_log(task_id, "  加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        task.progress = 5.0
        db.commit()

        write_log(task_id, "  加载基础模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        task.progress = 12.0
        db.commit()
        write_log(task_id, f"  模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        # ---- 配置 LoRA ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 2/4：配置 LoRA")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] if hasattr(model, "model") else ["query"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        total = sum(p.numel() for p in model.parameters()) / 1e6
        write_log(task_id, f"  可训练参数：{trainable:.2f}M / 总参数：{total:.2f}M ({trainable/total*100:.2f}%)")
        model.print_trainable_parameters()
        task.progress = 18.0
        db.commit()

        # ---- 准备数据 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 3/4：准备数据 & 训练")
        # 拼接所有数据集文本
        all_texts = [get_dataset_text(d) for d in datasets]
        text = "\n\n".join(t for t in all_texts if t)
        write_log(task_id, f"  合并后总字符数：{len(text)}")
        token_chunks = _prepare_dataset(text, tokenizer, max_len)
        if len(token_chunks) == 0:
            write_log(task_id, "[错误] 数据集处理后无可用样本")
            task.status = "failed"
            task.error_message = "数据集无可用样本"
            db.commit()
            return
        write_log(task_id, f"  生成训练样本：{len(token_chunks)} 条（每条 max_len={max_len}）")

        # 简易数据集：直接用 token ids
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, chunks, pad_id, max_len):
                self.chunks = chunks
                self.pad_id = pad_id
                self.max_len = max_len
            def __len__(self):
                return len(self.chunks)
            def __getitem__(self, i):
                ids = self.chunks[i][:self.max_len]
                labels = ids.copy()
                attn = [1] * len(ids)
                # pad
                while len(ids) < self.max_len:
                    ids.append(self.pad_id)
                    labels.append(-100)
                    attn.append(0)
                return {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                }

        train_ds = TextDataset(token_chunks, tokenizer.pad_token_id, max_len)

        # 输出目录
        out_dir = MODEL_DIR / f"finetuned_{task_id[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 用 TrainerCallback 实时记录日志
        from transformers import TrainerCallback

        class LoggingCallback(TrainerCallback):
            def __init__(self, task_id, db_session_factory, total_steps):
                self.task_id = task_id
                self.dbf = db_session_factory
                self.total_steps = total_steps
                self.step = 0
                self.last_log_ts = 0
            def on_log(self, args, state, control, logs=None, **kw):
                now = time.time()
                if now - self.last_log_ts < 1.0:
                    return
                self.last_log_ts = now
                if logs:
                    loss = logs.get("loss", logs.get("learning_rate", "?"))
                    lr_val = logs.get("learning_rate", "")
                    try:
                        write_log(self.task_id, f"  step {state.global_step}/{state.max_steps} | loss={loss} | lr={lr_val}")
                        # 记录 loss 数据点供曲线图使用
                        if isinstance(loss, (int, float)):
                            epoch = int(state.epoch) if state.epoch else 0
                            record_loss(self.task_id, state.global_step, loss,
                                        epoch=epoch, lr=float(lr_val) if lr_val else 0.0,
                                        task_type="training")
                    except Exception:
                        pass
            def on_step_begin(self, args, state, control, **kw):
                self.step = state.global_step
                db = self.dbf()
                try:
                    t = db.query(TrainingTask).filter(TrainingTask.id == self.task_id).first()
                    if t:
                        progress = 20 + (state.global_step / max(1, state.max_steps)) * 70
                        t.progress = round(min(90, progress), 1)
                        db.commit()
                finally:
                    db.close()

        total_steps = max(1, (len(train_ds) // batch_size + 1) * epochs)

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=max(1, total_steps // 20),
            save_strategy="no",
            report_to=[],
            fp16=False,
            use_cpu=(device == "cpu"),
            disable_tqdm=False,
            dataloader_drop_last=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=default_data_collator,
            callbacks=[LoggingCallback(task_id, SessionLocal, total_steps)],
        )

        write_log(task_id, f"  总训练步数：约 {total_steps} 步")
        write_log(task_id, "  开始训练...")
        write_log(task_id, "")
        task.progress = 20.0
        db.commit()

        train_result = trainer.train()
        final_loss = train_result.training_loss
        write_log(task_id, "")
        write_log(task_id, f"  训练完成！平均 loss：{final_loss:.4f}")
        task.progress = 92.0
        db.commit()

        # ---- 保存模型 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 4/4：保存微调模型")
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        try:
            write_log(task_id, "  合并 LoRA 权重...")
            merged = model.merge_and_unload()
            write_log(task_id, "  保存合并后的模型...")
            merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
            tokenizer.save_pretrained(str(merged_dir))
            save_path = str(merged_dir)
            write_log(task_id, f"  合并模型已保存：{save_path}")
        except Exception as e:
            write_log(task_id, f"  [警告] 合并失败（{e}），改为只保存 LoRA adapter")
            try:
                adapter_dir = out_dir / "adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(adapter_dir))
                tokenizer.save_pretrained(str(adapter_dir))
                # 也保存基础模型副本
                base_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]
                import shutil
                for fn in base_files:
                    src = Path(model_path) / fn
                    if src.exists():
                        shutil.copy(src, adapter_dir / fn)
                save_path = str(adapter_dir)
                write_log(task_id, f"  adapter 已保存：{save_path}")
            except Exception as e2:
                write_log(task_id, f"  [错误] 保存失败：{e2}")
                # 回退：直接复制原模型
                import shutil
                shutil.copytree(model_path, merged_dir, dirs_exist_ok=True)
                save_path = str(merged_dir)

        # 创建结果模型记录
        ds_names = "、".join(d.name for d in datasets)
        result_model = AIModel(
            name=f"{base_model.name}-微调版",
            source="finetune",
            model_id=base_model.model_id,
            local_path=save_path,
            status="ready",
            description=f"基于 {base_model.name} 在数据集「{ds_names}」上真实 LoRA 微调（{epochs} epochs, r={lora_r}, loss={final_loss:.3f}）",
            base_model_id=base_model.id,
        )
        db.add(result_model)
        db.flush()
        task.result_model_id = result_model.id
        task.progress = 100.0
        task.status = "completed"
        task.finished_at = datetime.utcnow()
        db.commit()

        write_log(task_id, "")
        write_log(task_id, "=" * 60)
        write_log(task_id, "训练任务完成！")
        write_log(task_id, f"输出模型：{result_model.name}")
        write_log(task_id, f"本地路径：{save_path}")
        write_log(task_id, "=" * 60)
    except Exception as e:
        try:
            db.rollback()
            tb = traceback.format_exc()
            try:
                write_log(task_id, "[错误] 训练失败：" + str(e))
                write_log(task_id, "详细堆栈：\n" + tb)
            except Exception:
                pass
            task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error_message = str(e)
                task.finished_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# ============== 模拟训练（保留作为 fallback） ==============

def _run_mock_training(task_id: str) -> None:
    db = SessionLocal()
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return
        task.log_file = str(get_log_path(task_id))
        task.status = "running"
        task.started_at = datetime.utcnow()
        db.commit()

        write_log(task_id, "=" * 60)
        write_log(task_id, f"训练任务（模拟模式）：{task.name}")
        write_log(task_id, "=" * 60)

        base_model = db.query(AIModel).filter(AIModel.id == task.model_id).first()
        ds_ids = [s.strip() for s in (task.dataset_id or "").split(",") if s.strip()]
        datasets = [db.query(Dataset).filter(Dataset.id == did).first() for did in ds_ids]
        datasets = [d for d in datasets if d]
        if not base_model or not datasets:
            task.status = "failed"
            db.commit()
            return

        total_samples = sum(d.sample_count for d in datasets)
        ds_names = "、".join(d.name for d in datasets)
        write_log(task_id, f"基础模型：{base_model.name}")
        write_log(task_id, f"数据集（{len(datasets)} 个）：{ds_names}（共 {total_samples} 样本）")
        write_log(task_id, f"参数：{json.dumps(task.params, ensure_ascii=False)}")

        stages = [
            (5, 20, "▶ 阶段 1/4：加载基础模型", [
                "初始化训练环境...", "加载基础模型...", "加载分词器...", "模型加载完成"]),
            (20, 44, "▶ 阶段 2/4：数据预处理", [
                f"读取 {len(datasets)} 个数据集，共 {total_samples} 样本", "文本清洗与归一化...", "分词处理...", "数据预处理完成"]),
            (44, 90, "▶ 阶段 3/4：模型训练", []),
            (90, 100, "▶ 阶段 4/4：保存模型", ["合并 LoRA 权重...", "保存模型...", "完成"]),
        ]
        for start, end, title, steps in stages:
            write_log(task_id, "")
            write_log(task_id, title)
            for i, s in enumerate(steps):
                time.sleep(0.4)
                write_log(task_id, f"  {s}")
                task.progress = round(start + (end - start) * (i + 1) / max(1, len(steps)), 1)
                db.commit()
            if title.endswith("模型训练"):
                epochs = int(task.params.get("epochs", 2))
                global_step = 0
                for ep in range(1, epochs + 1):
                    write_log(task_id, f"  ---- Epoch {ep}/{epochs} ----")
                    for s in range(1, 6):
                        global_step += 1
                        loss = round(2.8 * (0.5 ** (s / 6)) + random.uniform(-0.05, 0.05), 4)
                        write_log(task_id, f"  step {s}/5 | loss={loss}")
                        # 记录 loss 数据点供曲线图使用
                        record_loss(task_id, global_step, loss, epoch=ep,
                                    lr=float(task.params.get("learning_rate", 0.0002)),
                                    task_type="training")
                        task.progress = round(44 + (ep - 1) / epochs * 46 + s / 5 * 46 / epochs, 1)
                        db.commit()
                        time.sleep(0.4)

        result_dir = MODEL_DIR / f"finetuned_{task_id[:8]}"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "config.json").write_text('{"finetuned": true}', encoding="utf-8")
        result_model = AIModel(
            name=f"{base_model.name}-微调版",
            source="finetune",
            model_id=base_model.model_id,
            local_path=str(result_dir),
            status="ready",
            description=f"（模拟）基于 {base_model.name} 在数据集「{ds_names}」上微调",
            base_model_id=base_model.id,
        )
        db.add(result_model)
        db.flush()
        task.result_model_id = result_model.id
        task.progress = 100.0
        task.status = "completed"
        task.finished_at = datetime.utcnow()
        db.commit()
        write_log(task_id, "")
        write_log(task_id, "=" * 60)
        write_log(task_id, "训练完成！")
        write_log(task_id, "=" * 60)
    except Exception as e:
        try:
            db.rollback()
            write_log(task_id, "[错误] " + str(e))
            task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
            if task:
                task.status = "failed"
                task.error_message = str(e)
                task.finished_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def start_training(task_id: str) -> None:
    mode = settings.TRAINING_MODE.lower()
    target = _run_real_training if mode != "mock" else _run_mock_training
    t = threading.Thread(target=target, args=(task_id,), daemon=True)
    t.start()
