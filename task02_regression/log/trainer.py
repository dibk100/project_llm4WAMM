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

    def _save_checkpoint(self, model, trial, metrics=None):
        epoch = int(self.state.epoch or 0)
        checkpoint_folder = f"checkpoint-epoch-{epoch}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        self.save_model(output_dir=output_dir)
        self._save_optimizer_and_scheduler(output_dir)
        self._save_rng_state(output_dir)

        if metrics is not None:
            print("metrics가 없는 경우가 있어?")
            self.log_metrics("eval", metrics)
            self.save_metrics("eval", metrics)

        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    def save_model(self, output_dir=None, _internal_call=False):                        # output_dir 이거 때문이었음...
        print("💾💾 [CustomTrainer.save_model()] 호출됨!")  # 반드시 보여야 함
        
        epoch = int(self.state.epoch or 0)
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"💾💾 수동 저장 호출 - 모델 저장: {output_dir}")
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        # super().save_model(output_dir)                                                # 수동 저장 안되서 정말 사람 힘들게 한다.. 

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