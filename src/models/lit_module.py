import torch, torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

class LitClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, lr=3e-4, weight_decay=1e-2, scheduler="cosine", warmup_epochs=1):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc   = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc  = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1    = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        if stage == "train":
            self.train_acc.update(preds, y)
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == "val":
            self.val_acc.update(preds, y)
            self.val_f1.update(preds, y)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.test_acc.update(preds, y)
            self.log("test/loss", loss, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): self._step(batch, "val")
    def test_step(self, batch, batch_idx): self._step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=False)
        self.val_acc.reset(); self.val_f1.reset()

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_acc.compute(), prog_bar=True)
        self.test_acc.reset()
