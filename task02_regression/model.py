from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

class BertRegressionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name  # 저장용
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return self.regressor(cls_output)

    def get_model(self):
        return self

    def get_tokenizer(self):
        return self.tokenizer
