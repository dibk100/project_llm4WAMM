import torch.nn as nn
from transformers import AutoModel

class BertClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        return self.classifier(pooled_output)
