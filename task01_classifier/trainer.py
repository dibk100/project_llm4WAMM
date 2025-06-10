from transformers import Trainer
import torch
import torch.nn.functional as F
from transformers import TrainerCallback
import wandb
import os

class CustomTrainer(Trainer):
    def __init__(self, *args, binary_bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.binary_bool = binary_bool

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if self.binary_bool:
            if logits.size(1) == 1:
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            elif logits.size(1) == 2:
                labels = labels.view(-1).long()  # 꼭 1D long tensor
                loss = F.cross_entropy(logits, labels)
            else:
                raise ValueError(f"Unexpected logits shape for binary classification: {logits.shape}")
        else:
            # 멀티라벨 분류 (multi-hot labels)
            labels = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    def _save_checkpoint(self, model, trial, metrics=None):
        # ✅ epoch 기준 디렉토리명
        epoch = int(self.state.epoch or 0)
        checkpoint_folder = f"checkpoint-epoch-{epoch}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # ✅ 모델 저장 (내부 호출임을 명시)
        self.save_model(output_dir=output_dir, _internal_call=True)

        # ✅ 옵티마이저, 스케줄러, RNG 상태 저장
        self._save_optimizer_and_scheduler(output_dir)
        self._save_rng_state(output_dir)

        # ✅ 메트릭 및 state 저장
        if metrics is not None:
            self.log_metrics("eval", metrics)
            self.save_metrics("eval", metrics)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    def save_model(self, output_dir=None, _internal_call=False):
        epoch = int(self.state.epoch or 0)
        output_dir = output_dir or os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        #if _internal_call :
        #    print(f"내부 저장 호출 - 디렉토리 생성됨: {output_dir}")
        #    return

        print(f"수동 저장 호출 - 모델 저장: {output_dir}")
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            print("\n***** Val Results *****")
            for key, value in metrics.items():
                print(f"  {key} = {value:.4f}")
        return control

class WandbEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch은 float이지만 소수점 이하 버리고 int 처리 가능
        epoch = int(state.epoch)
        metrics = kwargs.get("metrics")
        if metrics:
            wandb.log(metrics, step=epoch)