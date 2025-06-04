from transformers import BertForSequenceClassification, BertTokenizer

class BertRegressionModel:
    def __init__(self, model_name='bert-base-uncased', device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="regression"
        ).to(self.device)
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
