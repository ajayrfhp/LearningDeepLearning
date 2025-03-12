import d2l.torch as d2l
import torch


class CustomTrainer(d2l.Trainer):
    def __init__(
        self,
        max_epochs,
        num_gpus=0,
        gradient_clip_val=0,
        rank=0,
        title="",
        plot_every_n_steps=10,
    ):
        super().__init__(max_epochs, num_gpus, gradient_clip_val)
        self.save_hyperparameters()
        self.train_dataloader = None
        self.val_dataloader = None
        self.num_train_batches = None
        self.num_val_batches = None
        self.model = None
        self.optimizer = None
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.epoch = 0
        if self.num_gpus > 0:
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

    def prepare_batch(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch["image"]
        y = batch["label"]
        return (x.to(self.device), y.to(self.device))

    def prepare_data(self, data: d2l.DataModule) -> None:
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = len(self.val_dataloader)

    def prepare_model(self, model: d2l.Module) -> None:
        model.trainer = self
        self.model = model
        self.model = self.model.to(self.device)

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optimizer = model.configure_optimizers()
        for _ in range(self.max_epochs):
            self.fit_epoch()
            self.epoch += 1
            self.model.display_metrics(self.num_train_batches, self.num_val_batches)

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            batch = self.prepare_batch(batch)
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward()
            self.optimizer.step()
        self.model.eval()
        for batch in self.val_dataloader:
            batch = self.prepare_batch(batch)
            self.model.validation_step(batch)
