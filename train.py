import torch
from transformers import get_scheduler
from torch.optim import AdamW

def train(model, dataloader, optimizer, device, num_epochs=3, lr=5e-5):
    model.to(device)
    model.train()
    
    # criterion 어디서 정의

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * num_epochs
    )

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
