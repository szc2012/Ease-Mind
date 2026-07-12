"""模型蒸馏服务

知识蒸馏（Knowledge Distillation）：
- 教师模型（大模型）对输入生成 logits（软标签）
- 学生模型（小模型）同时学习软标签和真实标签
- 损失 = alpha * KL(teacher_soft, student_soft) + (1-alpha) * CE(student, hard_labels)

通过 settings.TRAINING_MODE 控制：
- "real": 真实蒸馏
- "mock": 模拟流程
"""
import time
import json
import random
import threading
import traceback
from datetime import datetime
from pathlib import Path

from config import settings, LOG_DIR, MODEL_DIR
from database import SessionLocal
from models import DistillationTask, AIModel, Dataset
from services.dataset_service import get_dataset_text
from services.loss_service import record_loss
from schemas import DISTILL_PARAMS_DEFAULTS


# 参数定义（供前端展示）
DISTILL_PARAMS_CONFIG = {
    "temperature": {"label": "温度（Temperature）", "default": 2.0, "min": 1.0, "max": 10.0, "desc": "软化教师模型输出的 logits，越大越软"},
    "alpha": {"label": "蒸馏权重（Alpha）", "default": 0.5, "min": 0.0, "max": 1.0, "desc": "蒸馏损失占比，0=只用硬标签，1=只用软标签"},
    "epochs": {"label": "训练轮数", "default": 2, "min": 1, "max": 50},
    "batch_size": {"label": "批大小", "default": 2, "min": 1, "max": 16},
    "learning_rate": {"label": "学习率", "default": 0.0002, "min": 0.00001, "max": 0.01},
    "max_seq_length": {"label": "最大序列长度", "default": 256, "min": 128, "max": 1024},
}


def get_log_path(task_id: str) -> Path:
    return LOG_DIR / f"distill_{task_id}.log"


def write_log(task_id: str, message: str) -> None:
    path = get_log_path(task_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def read_log(task_id: str) -> str:
    path = get_log_path(task_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_chunks(text: str, tokenizer, max_len: int) -> list:
    """把长文本切成 max_len 的片段作为训练样本"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    samples = []
    step = max_len - 16
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_len - 16]
        if len(chunk) >= 32:
            samples.append(chunk)
        i += step
    return samples


def _load_model(model_path: str, device: str, dtype):
    """加载模型与 tokenizer"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return model, tok


def _run_real_distillation(task_id: str) -> None:
    """真实知识蒸馏"""
    db = SessionLocal()
    try:
        task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
        if not task:
            return
        task.log_file = str(get_log_path(task_id))
        task.status = "running"
        task.started_at = datetime.utcnow()
        task.progress = 0.0
        db.commit()

        write_log(task_id, "=" * 60)
        write_log(task_id, f"模型蒸馏任务：{task.name}")
        write_log(task_id, f"参数：{json.dumps(task.params, ensure_ascii=False)}")
        write_log(task_id, "=" * 60)

        teacher = db.query(AIModel).filter(AIModel.id == task.teacher_model_id).first()
        student = db.query(AIModel).filter(AIModel.id == task.student_model_id).first()
        ds_ids = [s.strip() for s in (task.dataset_id or "").split(",") if s.strip()]
        datasets = [db.query(Dataset).filter(Dataset.id == did).first() for did in ds_ids]
        datasets = [d for d in datasets if d]

        if not teacher or not teacher.local_path:
            write_log(task_id, "[错误] 教师模型未就绪")
            task.status = "failed"
            task.error_message = "教师模型未就绪"
            task.finished_at = datetime.utcnow()
            db.commit()
            return
        if not student or not student.local_path:
            write_log(task_id, "[错误] 学生模型未就绪")
            task.status = "failed"
            task.finished_at = datetime.utcnow()
            db.commit()
            return
        if teacher.id == student.id:
            write_log(task_id, "[错误] 教师模型与学生模型不能相同")
            task.status = "failed"
            task.error_message = "教师与学生模型相同"
            task.finished_at = datetime.utcnow()
            db.commit()
            return
        if not datasets:
            write_log(task_id, "[错误] 数据集不存在")
            task.status = "failed"
            task.finished_at = datetime.utcnow()
            db.commit()
            return

        # 解析参数
        p = dict(DISTILL_PARAMS_DEFAULTS)
        p.update(task.params or {})
        temperature = float(p["temperature"])
        alpha = float(p["alpha"])
        epochs = int(p["epochs"])
        batch_size = int(p["batch_size"])
        lr = float(p["learning_rate"])
        max_len = int(p["max_seq_length"])

        write_log(task_id, f"教师模型：{teacher.name}")
        write_log(task_id, f"学生模型：{student.name}")
        write_log(task_id, f"数据集（{len(datasets)} 个）：")
        for d in datasets:
            write_log(task_id, f"  - {d.name}（{d.sample_count} 样本）")
        write_log(task_id, f"温度 T={temperature}, alpha={alpha}, epochs={epochs}, batch={batch_size}, lr={lr}")

        device = _get_device()
        import torch
        dtype = torch.float16 if device != "cpu" else torch.float32
        write_log(task_id, f"设备：{device}, dtype：{dtype}")

        # ---- 阶段 1：加载教师模型 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 1/4：加载教师模型")
        teacher_model, teacher_tok = _load_model(teacher.local_path, device, dtype)
        t_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
        write_log(task_id, f"  教师参数量：{t_params:.1f}M")
        task.progress = 10.0
        db.commit()

        # ---- 阶段 2：加载学生模型 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 2/4：加载学生模型")
        student_model, student_tok = _load_model(student.local_path, device, dtype)
        s_params = sum(p.numel() for p in student_model.parameters()) / 1e6
        write_log(task_id, f"  学生参数量：{s_params:.1f}M")
        write_log(task_id, f"  压缩比：{t_params/max(s_params,0.01):.2f}x")
        task.progress = 20.0
        db.commit()

        # ---- 阶段 3：准备数据 & 教师生成软标签 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 3/4：准备数据 & 生成教师软标签")
        all_texts = [get_dataset_text(d) for d in datasets]
        text = "\n\n".join(t for t in all_texts if t)
        write_log(task_id, f"  合并后总字符数：{len(text)}")

        # 用学生 tokenizer 切分数据（保证学生可学）
        chunks = _prepare_chunks(text, student_tok, max_len)
        if not chunks:
            write_log(task_id, "[错误] 数据集处理后无可用样本")
            task.status = "failed"
            task.error_message = "数据集无可用样本"
            db.commit()
            return
        write_log(task_id, f"  生成训练样本：{len(chunks)} 条")

        # 构造数据集
        class DistillDataset(torch.utils.data.Dataset):
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
                while len(ids) < self.max_len:
                    ids.append(self.pad_id)
                    labels.append(-100)
                    attn.append(0)
                return {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                }

        train_ds = DistillDataset(chunks, student_tok.pad_token_id, max_len)

        # 预计算教师 logits（用教师 tokenizer 编码相同样本）
        write_log(task_id, "  生成教师模型软标签...")
        teacher_logits_list = []
        teacher_model.eval()
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]
                # 用教师 tokenizer 重新编码（用学生 tokenizer 解码回文本，再用教师 tokenizer 编码）
                t_inputs = teacher_tok(
                    [student_tok.decode(c) for c in batch_chunks],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                ).to(device)
                t_logits = teacher_model(**t_inputs).logits  # [B, seq, vocab]
                # 按学生词表大小截断或对齐（简化：取学生词表大小）
                # 注意：教师与学生词表可能不同，这里取学生词表大小
                vocab_size = student_model.config.vocab_size
                if t_logits.size(-1) > vocab_size:
                    t_logits = t_logits[..., :vocab_size]
                elif t_logits.size(-1) < vocab_size:
                    # pad
                    pad_size = vocab_size - t_logits.size(-1)
                    pad = torch.full((*t_logits.shape[:-1], pad_size), -1e9, device=device, dtype=t_logits.dtype)
                    t_logits = torch.cat([t_logits, pad], dim=-1)
                teacher_logits_list.append(t_logits.cpu())
                if (i // batch_size) % 5 == 0:
                    write_log(task_id, f"    已生成 {min(i+batch_size, len(chunks))}/{len(chunks)} 样本的软标签")
        write_log(task_id, "  教师软标签生成完成")
        task.progress = 35.0
        db.commit()

        # 释放教师模型显存/内存
        del teacher_model
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        write_log(task_id, "  教师模型已卸载，释放显存")

        # ---- 阶段 4：蒸馏训练学生模型 ----
        write_log(task_id, "")
        write_log(task_id, "▶ 阶段 4/4：蒸馏训练学生模型")

        # 学生模型开启训练模式
        student_model.train()
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=0.01)

        kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        total_steps = max(1, (len(train_ds) // batch_size + 1) * epochs)
        global_step = 0
        log_interval = max(1, total_steps // 20)

        for epoch in range(1, epochs + 1):
            write_log(task_id, f"  ---- Epoch {epoch}/{epochs} ----")
            for i in range(0, len(train_ds), batch_size):
                batch_samples = [train_ds[j] for j in range(i, min(i+batch_size, len(train_ds)))]
                input_ids = torch.stack([s["input_ids"] for s in batch_samples]).to(device)
                labels = torch.stack([s["labels"] for s in batch_samples]).to(device)
                attn_mask = torch.stack([s["attention_mask"] for s in batch_samples]).to(device)

                # 学生 logits
                student_logits = student_model(input_ids=input_ids, attention_mask=attn_mask).logits

                # 教师 logits（对应 batch）
                t_batch_idx = i // batch_size
                if t_batch_idx < len(teacher_logits_list):
                    teacher_logits = teacher_logits_list[t_batch_idx].to(device)
                    # 对齐序列长度
                    if teacher_logits.size(1) != student_logits.size(1):
                        min_len = min(teacher_logits.size(1), student_logits.size(1))
                        teacher_logits = teacher_logits[:, :min_len, :]
                        student_logits_adj = student_logits[:, :min_len, :]
                        labels_adj = labels[:, :min_len]
                    else:
                        student_logits_adj = student_logits
                        labels_adj = labels
                else:
                    teacher_logits = None
                    student_logits_adj = student_logits
                    labels_adj = labels

                # 蒸馏损失：KL(teacher_soft || student_soft)
                if teacher_logits is not None and alpha > 0:
                    soft_student = torch.log_softmax(student_logits_adj / temperature, dim=-1)
                    soft_teacher = torch.softmax(teacher_logits / temperature, dim=-1)
                    distill_loss = kl_loss_fn(soft_student, soft_teacher) * (temperature ** 2)
                else:
                    distill_loss = torch.tensor(0.0, device=device)

                # 硬标签损失：CE(student, labels)
                ce_loss = ce_loss_fn(student_logits_adj.view(-1, student_logits_adj.size(-1)), labels_adj.view(-1))

                loss = alpha * distill_loss + (1 - alpha) * ce_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

                global_step += 1
                if global_step % log_interval == 0:
                    write_log(task_id, f"  step {global_step}/{total_steps} | loss={loss.item():.4f} | distill={distill_loss.item():.4f} | ce={ce_loss.item():.4f}")
                    # 记录 loss 数据点供曲线图使用
                    record_loss(task_id, global_step, loss.item(),
                                epoch=epoch, lr=lr,
                                extra={"distill_loss": distill_loss.item(), "ce_loss": ce_loss.item()},
                                task_type="distillation")

                # 更新进度
                progress = 35 + (global_step / total_steps) * 55
                task.progress = round(min(90, progress), 1)
                db.commit()

        write_log(task_id, f"  蒸馏训练完成，总步数 {global_step}")
        task.progress = 92.0
        db.commit()

        # ---- 保存蒸馏后的学生模型 ----
        out_dir = MODEL_DIR / f"distilled_{task_id[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        try:
            write_log(task_id, "  保存蒸馏后的学生模型...")
            student_model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="2GB")
            student_tok.save_pretrained(str(merged_dir))
            save_path = str(merged_dir)
            write_log(task_id, f"  模型已保存：{save_path}")
        except Exception as e:
            write_log(task_id, f"  [警告] 保存失败：{e}")
            import shutil
            shutil.copytree(student.local_path, merged_dir, dirs_exist_ok=True)
            save_path = str(merged_dir)

        # 创建结果模型记录
        ds_names = "、".join(d.name for d in datasets)
        result_model = AIModel(
            name=f"{student.name}-蒸馏版",
            source="distill",
            model_id=student.model_id,
            local_path=save_path,
            status="ready",
            description=f"由教师模型「{teacher.name}」蒸馏学生模型「{student.name}」（T={temperature}, alpha={alpha}, {epochs} epochs）｜数据集：{ds_names}",
            base_model_id=student.id,
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
        write_log(task_id, "蒸馏任务完成！")
        write_log(task_id, f"输出模型：{result_model.name}")
        write_log(task_id, "=" * 60)
    except Exception as e:
        db.rollback()
        tb = traceback.format_exc()
        write_log(task_id, "[错误] 蒸馏失败：" + str(e))
        write_log(task_id, "详细堆栈：\n" + tb)
        task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
        if task:
            task.status = "failed"
            task.error_message = str(e)
            task.finished_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def _run_mock_distillation(task_id: str) -> None:
    """模拟蒸馏流程"""
    db = SessionLocal()
    try:
        task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
        if not task:
            return
        task.log_file = str(get_log_path(task_id))
        task.status = "running"
        task.started_at = datetime.utcnow()
        db.commit()

        write_log(task_id, "=" * 60)
        write_log(task_id, f"模型蒸馏任务（模拟模式）：{task.name}")
        write_log(task_id, "=" * 60)

        teacher = db.query(AIModel).filter(AIModel.id == task.teacher_model_id).first()
        student = db.query(AIModel).filter(AIModel.id == task.student_model_id).first()
        ds_ids = [s.strip() for s in (task.dataset_id or "").split(",") if s.strip()]
        datasets = [db.query(Dataset).filter(Dataset.id == did).first() for did in ds_ids]
        datasets = [d for d in datasets if d]
        if not teacher or not student or not datasets:
            task.status = "failed"
            db.commit()
            return

        total_samples = sum(d.sample_count for d in datasets)
        ds_names = "、".join(d.name for d in datasets)
        write_log(task_id, f"教师模型：{teacher.name}")
        write_log(task_id, f"学生模型：{student.name}")
        write_log(task_id, f"数据集（{len(datasets)} 个）：{ds_names}（共 {total_samples} 样本）")
        write_log(task_id, f"参数：{json.dumps(task.params, ensure_ascii=False)}")

        stages = [
            (5, 20, "▶ 阶段 1/4：加载教师模型", ["初始化...", "加载教师模型...", "教师模型就绪"]),
            (20, 35, "▶ 阶段 2/4：加载学生模型", ["加载学生模型...", "学生模型就绪"]),
            (35, 50, "▶ 阶段 3/4：生成教师软标签", [f"教师模型对 {total_samples} 样本生成 logits...", "软标签生成完成"]),
            (50, 90, "▶ 阶段 4/4：蒸馏训练学生模型", []),
            (90, 100, "保存蒸馏模型", ["保存模型...", "完成"]),
        ]
        for start, end, title, steps in stages:
            write_log(task_id, "")
            write_log(task_id, title)
            for i, s in enumerate(steps):
                time.sleep(0.4)
                write_log(task_id, f"  {s}")
                task.progress = round(start + (end - start) * (i + 1) / max(1, len(steps)), 1)
                db.commit()
            if "蒸馏训练" in title:
                epochs = int(task.params.get("epochs", 2))
                global_step = 0
                for ep in range(1, epochs + 1):
                    write_log(task_id, f"  ---- Epoch {ep}/{epochs} ----")
                    for s in range(1, 6):
                        global_step += 1
                        distill_loss = round(2.0 * (0.6 ** (s / 6)) + random.uniform(-0.05, 0.05), 4)
                        ce_loss = round(2.5 * (0.5 ** (s / 6)) + random.uniform(-0.05, 0.05), 4)
                        total_loss = round(0.5 * distill_loss + 0.5 * ce_loss, 4)
                        write_log(task_id, f"  step {s}/5 | distill={distill_loss} | ce={ce_loss}")
                        record_loss(task_id, global_step, total_loss, epoch=ep,
                                    lr=float(task.params.get("learning_rate", 0.0002)),
                                    extra={"distill_loss": distill_loss, "ce_loss": ce_loss},
                                    task_type="distillation")
                        task.progress = round(50 + (ep - 1) / epochs * 40 + s / 5 * 40 / epochs, 1)
                        db.commit()
                        time.sleep(0.4)

        result_dir = MODEL_DIR / f"distilled_{task_id[:8]}"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "config.json").write_text('{"distilled": true}', encoding="utf-8")
        result_model = AIModel(
            name=f"{student.name}-蒸馏版",
            source="distill",
            model_id=student.model_id,
            local_path=str(result_dir),
            status="ready",
            description=f"（模拟）由 {teacher.name} 蒸馏 {student.name}",
            base_model_id=student.id,
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
        write_log(task_id, "蒸馏完成！")
        write_log(task_id, "=" * 60)
    except Exception as e:
        write_log(task_id, "[错误] " + str(e))
        task = db.query(DistillationTask).filter(DistillationTask.id == task_id).first()
        if task:
            task.status = "failed"
            task.error_message = str(e)
            db.commit()
    finally:
        db.close()


def start_distillation(task_id: str) -> None:
    mode = settings.TRAINING_MODE.lower()
    target = _run_real_distillation if mode != "mock" else _run_mock_distillation
    t = threading.Thread(target=target, args=(task_id,), daemon=True)
    t.start()
