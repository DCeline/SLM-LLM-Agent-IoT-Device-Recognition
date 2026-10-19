import torch
import torch.nn as nn
from transformers import RobertaModel
from torchcrf import CRF
from sklearn.metrics import classification_report

class BERT_CRF_NER(nn.Module):
    def __init__(self, roberta_model, num_tags, dropout_rate=0.5):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        # 移除 projection 层，直接使用 roberta 的 hidden_size
        self.dropout = nn.Dropout(dropout_rate)  # 增加 Dropout 层
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)  # batch_first=True 表示输入的第一个维度是 batch_size
        self._init_transition_constraints(num_tags)

    def _init_transition_constraints(self, num_tags):
        # 优化约束矩阵，使其更符合设备型号实体的 BIOE 标注规则
        # 将默认惩罚放宽（从 -2 -> -1），并允许 B->O 转移（单字实体需要）
        constraint_matrix = torch.full((num_tags, num_tags), -1.0)
        # 假设标签顺序: 0=O, 1=B-DEVICE, 2=I-DEVICE, 3=E-DEVICE

        # O可以转到O或B-DEVICE
        constraint_matrix[0, 0] = 0      # O→O
        constraint_matrix[0, 1] = 0      # O→B-DEVICE

        # B-DEVICE只能转到I-DEVICE或E-DEVICE
        constraint_matrix[1, 2] = 2.0    # B-DEVICE→I-DEVICE（强激励）
        constraint_matrix[1, 3] = 1.0    # B-DEVICE→E-DEVICE（适度激励）
        # 允许单字实体后接 O（B->O），提高单字实体学习能力
        constraint_matrix[1, 0] = 0.5    # B-DEVICE→O（允许单字实体）

        # I-DEVICE只能转到I-DEVICE或E-DEVICE
        constraint_matrix[2, 2] = 0.5    # I-DEVICE→I-DEVICE（弱激励，防止过长实体）
        constraint_matrix[2, 3] = 2.0    # I-DEVICE→E-DEVICE（强激励）

        # E-DEVICE只能转到O或B-DEVICE
        constraint_matrix[3, 0] = 2.0    # E-DEVICE→O
        constraint_matrix[3, 1] = 0.5    # E-DEVICE→B-DEVICE（允许连续实体）

        self.crf.transitions = nn.Parameter(constraint_matrix)

    def forward(self, input_ids, attention_mask, labels=None):
        sequence_output = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        attention_mask = attention_mask.clone()
        attention_mask[:, 0] = 1

        if labels is not None:
            # 定义权重（CRF.forward 不接受 weights 参数），通过调整 emissions 来实现类别权重
            weights = torch.tensor([0.1, 1.0, 1.0, 1.0], device=emissions.device, dtype=emissions.dtype)  # O 的权重较低
            # 将权重作为对数偏置加到 emissions 上，等价于对某些标签降低/提高得分，从而影响 CRF 的损失和解码
            emissions = emissions + torch.log(weights).unsqueeze(0).unsqueeze(0)
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction="mean")
            return loss
        else:
            return emissions

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        ce_loss = nn.CrossEntropyLoss()(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[1, 2, 3], target_names=["B-DEVICE", "I-DEVICE", "E-DEVICE"])
    print(report)
