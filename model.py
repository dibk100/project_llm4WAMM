from transformers import AutoModelForSequenceClassification

def get_model(config):
    model_name = config['model_name']
    num_labels = len(config['labels'])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"  # 멀티라벨 분류임을 명시 : loss = BCEWithLogitsLoss()(logits, labels)
    )
    return model
