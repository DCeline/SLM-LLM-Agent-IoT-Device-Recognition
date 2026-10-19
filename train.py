import os
import json
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from dataset import DeviceNERDataset

def train_model(model, tokenizer, train_json_path, val_json_path=None, epochs=10, batch_size=16, lr=5e-5,
                model_save_path="device_detector_model.pth", seqlen=128, stride=64, logs_dir="logs",
                val_batch_size=32, early_stopping_patience=3, monitor_labels=[1,2,3]):
    """训练设备检测模型，支持验证集、early stopping 和保存最佳模型
    :param val_json_path: 验证集路径（若为 None 则不做验证与 early stopping）
    :param early_stopping_patience: 验证集无提升时的容忍轮数
    :param monitor_labels: 用于计算召回的标签集合（B/I/E）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = DeviceNERDataset(train_json_path, tokenizer, seqlen=seqlen, stride=stride, debug=False)
    # 计算每个样本是否包含实体，用于加权采样（过采样包含实体的样本）
    sample_weights = []
    for s in train_dataset.samples:
        labels = s['labels']
        sample_weights.append(5.0 if (labels != 0).any().item() else 1.0)

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False
    )

    # 如果提供了验证集路径，创建验证 DataLoader（不打乱）
    val_loader = None
    if val_json_path:
        val_dataset = DeviceNERDataset(val_json_path, tokenizer, seqlen=seqlen, stride=stride, debug=False)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    # 日志和早停准备
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_epoch_records = []

    best_monitor_value = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    best_model_path = model_save_path + ".best.pth"

    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_labels = [], []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            if hasattr(loss, "dim") and loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            with torch.no_grad():
                emissions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                weights = torch.tensor([0.1, 1.0, 1.0, 1.0], device=emissions.device, dtype=emissions.dtype)
                emissions_biased = emissions + torch.log(weights).unsqueeze(0).unsqueeze(0)
                preds = model.crf.decode(emissions_biased, mask=attention_mask.bool())

                for i in range(labels.shape[0]):
                    seq_len = int(attention_mask[i].sum().item())
                    all_preds.extend([int(x) for x in preds[i][:seq_len]])
                    all_labels.extend([int(x) for x in labels[i][:seq_len].cpu().numpy()])

        avg_loss = total_loss / max(1, len(train_loader))
        y_true = all_labels
        y_pred = all_preds

        report_text = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=["O", "B-DEVICE", "I-DEVICE", "E-DEVICE"],
            zero_division=0
        )

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)

        epoch_record = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "accuracy": float(accuracy),
            "labels": {
                "O": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
                "B-DEVICE": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
                "I-DEVICE": {"precision": float(precision[2]), "recall": float(recall[2]), "f1": float(f1[2]), "support": int(support[2])},
                "E-DEVICE": {"precision": float(precision[3]), "recall": float(recall[3]), "f1": float(f1[3]), "support": int(support[3])},
            },
            "classification_report": report_text
        }
        per_epoch_records.append(epoch_record)

        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        print(report_text)

        # 如果有验证集，则在验证集上评估并执行 early stopping
        if val_loader is not None:
            model.eval()
            val_all_preds, val_all_labels = [], []
            with torch.no_grad():
                for vbatch in val_loader:
                    v_input_ids = vbatch['input_ids'].to(device)
                    v_attention_mask = vbatch['attention_mask'].to(device)
                    v_labels = vbatch['labels'].to(device)

                    v_emissions = model(input_ids=v_input_ids, attention_mask=v_attention_mask)
                    weights = torch.tensor([0.1, 1.0, 1.0, 1.0], device=v_emissions.device, dtype=v_emissions.dtype)
                    v_emissions_biased = v_emissions + torch.log(weights).unsqueeze(0).unsqueeze(0)
                    v_preds = model.crf.decode(v_emissions_biased, mask=v_attention_mask.bool())

                    for i in range(v_labels.shape[0]):
                        seq_len = int(v_attention_mask[i].sum().item())
                        val_all_preds.extend([int(x) for x in v_preds[i][:seq_len]])
                        val_all_labels.extend([int(x) for x in v_labels[i][:seq_len].cpu().numpy()])

            # 只关注 monitor_labels 的平均召回率
            _, val_recall, _, _ = precision_recall_fscore_support(
                val_all_labels, val_all_preds, labels=monitor_labels, zero_division=0
            )
            monitor_value = float(sum(val_recall) / max(1, len(val_recall)))

            print(f"[Validation] epoch={epoch+1} monitor_recall(mean over {monitor_labels})={monitor_value:.4f}")

            # 若改善则保存最佳模型并重置 patience 计数
            if monitor_value > best_monitor_value:
                best_monitor_value = monitor_value
                best_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"[Validation] New best model saved to {best_model_path} (monitor={monitor_value:.4f})")
            else:
                no_improve_epochs += 1
                print(f"[Validation] No improvement for {no_improve_epochs} epoch(s)")

            model.train()

            # 检查 early stopping
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered. Best epoch: {best_epoch}, best_monitor={best_monitor_value:.4f}")
                break

    # 训练结束后保存最后模型与日志
    torch.save(model.state_dict(), model_save_path)
    print(f"模型训练完成并已保存到 {model_save_path}. 最佳模型: {best_model_path} (如果存在)")

    json_path = os.path.join(logs_dir, f"train_metrics_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(per_epoch_records, jf, ensure_ascii=False, indent=4)
    print(f"训练指标已保存到 {json_path}")

    csv_path = os.path.join(logs_dir, f"train_metrics_{timestamp}.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
        writer = csv.writer(cf)
        header = [
            "epoch", "avg_loss", "accuracy",
            "O_precision", "O_recall", "O_f1", "O_support",
            "B_precision", "B_recall", "B_f1", "B_support",
            "I_precision", "I_recall", "I_f1", "I_support",
            "E_precision", "E_recall", "E_f1", "E_support"
        ]
        writer.writerow(header)
        for rec in per_epoch_records:
            labels = rec["labels"]
            row = [
                rec["epoch"], rec["avg_loss"], rec["accuracy"],
                labels["O"]["precision"], labels["O"]["recall"], labels["O"]["f1"], labels["O"]["support"],
                labels["B-DEVICE"]["precision"], labels["B-DEVICE"]["recall"], labels["B-DEVICE"]["f1"], labels["B-DEVICE"]["support"],
                labels["I-DEVICE"]["precision"], labels["I-DEVICE"]["recall"], labels["I-DEVICE"]["f1"], labels["I-DEVICE"]["support"],
                labels["E-DEVICE"]["precision"], labels["E-DEVICE"]["recall"], labels["E-DEVICE"]["f1"], labels["E-DEVICE"]["support"]
            ]
            writer.writerow(row)
    print(f"训练指标 CSV 已保存到 {csv_path}")

    final_txt_path = os.path.join(logs_dir, f"final_classification_report_{timestamp}.txt")
    with open(final_txt_path, "w", encoding="utf-8") as tf:
        tf.write(per_epoch_records[-1]["classification_report"])
    print(f"最后一轮 classification_report 已保存到 {final_txt_path}")

    return per_epoch_records, {"best_epoch": best_epoch, "best_monitor": best_monitor_value, "best_model_path": best_model_path}