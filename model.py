import torch.nn as nn
from transformers import AutoModel

class BertForMultiLabelClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
