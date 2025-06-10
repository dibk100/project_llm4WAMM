from transformers import AutoModelForSequenceClassification

def get_model(config):
    model_name = config['model_name']
    binary_bool = config['binary_bool']

    if binary_bool:
        num_labels = 2
        problem_type = "single_label_classification"  # 이진 분류용
    else:
        num_labels = len(config['labels'])
        problem_type = "multi_label_classification"  # 멀티라벨용

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type
    )

    return model

# single_label_classification → CrossEntropyLoss 사용
# multi_label_classification → BCEWithLogitsLoss 사용
