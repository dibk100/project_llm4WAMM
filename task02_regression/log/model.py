from transformers import BertModel, BertTokenizer,RobertaTokenizer, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
import os

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

    def forward(self, input_ids, attention_mask,labels=None):
        """
        trainer 사용시, forward 수정 필요
        Huggingface Trainer는 내부적으로 평가할 때 모델의 forward()매서드라 labels 인자를 받고 labels가 있으면 loss를 반환함.
        모델이 labels를 받지 않는다면 loss를 반환하지 않아서 Trainer를 평가할 때 loss 계산을 못하고 metrics함수 호출하지 않는다고 함.
        
        지금 데이터에서 labels가 반환되지 않은 이슈가 있는건가?
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        # return self.regressor(cls_output)
        logits = self.regressor(cls_output).view(-1)
        # logits = self.regressor(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits, labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def get_model(self):
        return self

    def get_tokenizer(self):
        return self.tokenizer

def load_best_model(config, best_model_path, device):
    model_wrapper = BertRegressionModel(config['model_name'])
    model = model_wrapper.get_model()
    
    state_dict = torch.load(os.path.join(best_model_path, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    tokenizer = model_wrapper.get_tokenizer()
    return model, tokenizer


