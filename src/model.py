import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from functools import partial
from dino3d import VisionTransformer3D  # Import your actual model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import classification_report, accuracy_score
from lightning.pytorch.utilities.types import LRSchedulerConfig


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.layer(x)


class VisionTransformerEEG(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-4, pretrained_ckpt_path=None, freeze=False, class_weights=None):
        super().__init__()
        
        # === [ADDED] Init prediction storage for validation reporting ===
        self._val_preds = []
        self._val_labels = []
        self.val_epoch_acc = []

        self.save_hyperparameters()

        self.backbone = VisionTransformer3D(
            img_size=(238, 19, 400),
            patch_size=(14, 3, 40),
            in_chans=1,
            num_classes=num_classes,  # Head is defined separately
            embed_dim= 384,
            depth=6,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-2),
        )
        
        # after self.backbone = VisionTransformer3D(...)
        
        self.age_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # Combine DINO ViT 384 with age (16) → 400
        self.classifier = nn.Sequential(
            
         
            nn.Linear(384+16, 512),   
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.25),
            ResidualBlock(512, dropout=0.25),
           
            
            

            nn.Linear(512, num_classes)  # Final output layer
        )

        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


        #self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.learning_rate = learning_rate
        
        # Load pretrained weights only if not resuming from a checkpoint
        if pretrained_ckpt_path and not getattr(self, '_loaded_from_checkpoint', False):
            self.backbone.init_weights(pretrained=pretrained_ckpt_path, bootstrap_method="centering")

        """
        # Load pretrained weights
        if pretrained_ckpt_path:
            self.backbone.init_weights(pretrained=pretrained_ckpt_path, bootstrap_method="centering")
        """
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        #print("[INFO] 🔒 Freezing full backbone...")
        trainable_keywords = [
            "patch_embed", "cls_token", "pos_embed", "norm"
        ]

        for name, param in self.backbone.named_parameters():
            if any(kw in name for kw in trainable_keywords):
                param.requires_grad = True
                print(f"✅ {name} → trainable")
            else:
                param.requires_grad = False
                print(f"🧊 {name} → frozen")
        
        for name, param in self.classifier.named_parameters():
            param.requires_grad = True
            print(f"✅ classifier.{name} → trainable")
        """
        for name, param in self.backbone.head.named_parameters():
            param.requires_grad = True
            print(f"✅ classifier.{name} → trainable")
        #self.backbone.eval()
        """
    def forward(self, x, age):
        x = self.backbone(x)   
        age = age.float().unsqueeze(1).to(dtype=self.dtype)
        # [B, 384]
        age_feat = self.age_embed(age)      # [B, 16]
        x = torch.cat([x, age_feat], dim=1) # [B, 400]
        return self.classifier(x)
    


   

    
    def training_step(self, batch, batch_idx):
        x, y, age = batch
        logits = self(x, age)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss, "acc": acc}
    """
    def mixup(self, x, y, alpha=0.2):
        # Ensure alpha is a float
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.flatten()[0].item()

        lam = np.random.beta(float(alpha), float(alpha))
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        x_mix = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return x_mix, y_a, y_b, lam



    def training_step(self, batch, batch_idx):
        x, y, age = batch

        # Use your class method safely
        x, y_a, y_b, lam = self.mixup(x, y)

        logits = self(x, age)

        loss = lam * self.loss_fn(logits, y_a) + (1 - lam) * self.loss_fn(logits, y_b)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss, "acc": acc}
    """

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f"[DEBUG] model.training: {self.training}")
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Dropout):
                    print(f"[DEBUG] Dropout layer '{name}': training={module.training}")

        x, y, age = batch
        logits = self(x, age)
        preds = logits.argmax(dim=1)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Save it for checkpoint hook
        self.current_val_acc = acc.item()  # ✅ current best val_
        #self._val_preds.append(preds.detach().cpu())
        #self._val_labels.append(y.detach().cpu())

        return {"val_loss": loss, "val_acc": acc}  

    def test_step(self, batch, batch_idx):
        
        x, y, age = batch
        logits = self(x, age)
        
        loss = self.loss_fn(logits, y)
        
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"test_loss": loss, "test_acc": acc}
    """
    def on_validation_epoch_start(self):
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
            print("🔒 Backbone set to eval() during validation")
    """
    
    """
    def on_validation_epoch_end(self):
        
        preds = torch.cat(self._val_preds, dim=0)
        labels = torch.cat(self._val_labels, dim=0)

        acc = accuracy_score(labels, preds)
        self.val_epoch_acc.append(acc)  # ✅ Store for plotting

        print("\n📊 Classification Report (Validation):")
        print(classification_report(labels, preds, target_names=["Class 0", "Class 1", "Class 2"]))
        print(f"✅ Validation Accuracy: {acc:.4f}")

        # Clear memory
        self._val_preds.clear()
        self._val_labels.clear()
    """
    
    def get_cls_embeddings(self, dataloader, device='cuda'):
        self.eval()
        self.to(device)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for x, y, age in dataloader:
                x, age = x.to(device), age.to(device)
                out = self.backbone(x)
                if out.dim() == 3:
                    out = out[:, 0]  # CLS token

                all_embeddings.append(out.cpu())
                all_labels.extend(y.cpu().numpy())

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        return all_embeddings, all_labels

    def configure_optimizers(self):
        embed_params = [
            p for n, p in self.backbone.named_parameters()
            if ("patch_embed" in n or "pos_embed" in n or "cls_token" in n) and p.requires_grad
        ]
        transformer_tail_params = [
            p for n, p in self.backbone.named_parameters()
            if ("blocks.10" in n or "blocks.11" in n or "norm" in n) and p.requires_grad
        ]
        classifier_params = list(self.classifier.parameters())

        optimizer = torch.optim.Adam([
            {"params": embed_params, "lr": 5e-5, "weight_decay": 0.05},
            {"params": transformer_tail_params, "lr": 1e-4, "weight_decay": 0.05},
            {"params": classifier_params, "lr": 1e-3, "weight_decay": 0.05}
        ])

        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=10)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs - 10)

        scheduler = SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[10]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint["val_acc"] = getattr(self, "current_val_acc", None)
        checkpoint["epoch"] = self.current_epoch  # Optional: Save the epoch number
