from transformers import AutoModelForSequenceClassification

def get_model(config):
    model_name = config['model_name']
    binary_bool = config['binary_bool']
    
    if binary_bool:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            problem_type="single_label_classification"  # 이진 분류면 single_label_classification
        )

    else :
        num_labels = len(config['labels'])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"  # 멀티라벨 분류임을 명시 : loss = BCEWithLogitsLoss()(logits, labels)
        )
        
    return model