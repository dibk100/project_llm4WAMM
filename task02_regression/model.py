from transformers import BertModel, BertTokenizer,RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn

class BertRegressionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name  # 저장용
        
        if model_name == "roberta-base" :
            self.bert = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
               
        else :
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.model_name.startswith("roberta"):
            cls_output = outputs.last_hidden_state[:, 0, :]  # RoBERTa는 pooler가 없음
        else:
            cls_output = outputs.pooler_output  # BERT는 pooler_output이 CLS임
        
        return self.regressor(cls_output).squeeze(-1)

    def get_model(self):
        return self

    def get_tokenizer(self):
        return self.tokenizer