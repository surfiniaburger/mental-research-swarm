from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def configure_optimizers(self):
    optimizer = AdamW(
        self.model.parameters(), 
        lr=self.args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=1e-2
    )

    scheduler_instance = CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=self.args.warmup_epochs + 256,      # Heuristic based on typical plateau size
        T_mult=2.0,                            # Double the period each restart
        eta_min=self.args.learning_rate * 0.1  # Decay floor
    )

    return {"optimizer": optimizer, "lr_scheduler": scheduler_instance}