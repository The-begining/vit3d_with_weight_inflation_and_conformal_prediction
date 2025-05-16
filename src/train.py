import os   
import torch
import numpy as np
import pytorch_lightning as pl
from utils import plot_logs
import torch.nn.functional as F
from model_500 import VisionTransformerEEG
from dataset_with_aug import get_dataloaders
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
from model_500 import visualize_attention_slices, overlay_attention_on_eeg  # make sure these functions are in model_500.py



    # --- Logger ---
def main():
    



    logger = CSVLogger(save_dir="eeg_classification/logs", name="my_run")
    # Restrict thread usage to avoid CPU oversubscription
    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    os.environ["MKL_NUM_THREADS"] = str(num_cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus)
    torch.set_num_threads(num_cpus)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_acc:.4f}",  # ← unique name for best
        verbose=True
    )
    # --- Paths ---
    pretrained_ckpt = None # "/home/shubham/D1/ViT/eeg_classification/dino/pretrained/dino_deitsmall8_pretrain.pth"
    train_dir = "/home/shubham/D1/caueeg_new_new/caueeg/train"
    val_dir = "/home/shubham/D1/caueeg_new_new/caueeg/val"
    test_dir = "/home/shubham/D1/caueeg_new_new/caueeg/test"
    resume_ckpt = "/home/shubham/D1/ViT/eeg_classification/eeg_classification/logs/my_run/version_218_70_nocheckpoint/checkpoints/best-checkpoint.ckpt"  
    # --- Hyperparams ---
    batch_size = 8
    num_classes = 3
    learning_rate = 5e-6
    max_epochs = 50

    # --- Data ---
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
       # class_pair=(1, 2)
    )
    
    #train_embeddings, train_labels = model.get_cls_embeddings(train_loader)
    # Step 1: Collect labels
    train_labels = []
    for _, labels, _ in train_loader.dataset:
        train_labels.append(labels)
    train_labels = np.array(train_labels)

    # Step 2: Create weights: only class 2 gets boosted
    classes_in_data = np.unique(train_labels)  # should be [0, 1]
    print("Classes found:", classes_in_data)
    class_weights_raw = compute_class_weight(class_weight='balanced', classes=classes_in_data, y=train_labels)
    class_weights = torch.tensor(class_weights_raw, dtype=torch.float32).to("cuda")
    print("Class weights →", dict(zip(classes_in_data, class_weights_raw)))

    """   
    
    # Step 1: Collect labels
    train_labels = []
    for _, labels, _ in train_loader.dataset:
        train_labels.append(labels)
    train_labels = np.array(train_labels)

    # Step 2: Compute class weights across all present labels
    unique_classes = np.unique(train_labels)
    print("Classes found:", unique_classes)

    class_weights_raw = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )

    class_weights = torch.tensor(class_weights_raw, dtype=torch.float32).to("cuda")
    print("Computed class weights:", dict(zip(unique_classes, class_weights_raw)))
    """
    """
    # --- Model ---
    model = VisionTransformerEEG(
            num_classes=num_classes,
            learning_rate=learning_rate,
            pretrained_ckpt_path=pretrained_ckpt,
            freeze=True,
            class_weights=class_weights
        )
    """
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        model = VisionTransformerEEG.load_from_checkpoint(resume_ckpt)
    else:
        print(f"Training from scratch with pretrained weights: {pretrained_ckpt}")
        model = VisionTransformerEEG(
            num_classes=num_classes,
            learning_rate=learning_rate,
            pretrained_ckpt_path=pretrained_ckpt,
            freeze=True,
            class_weights=class_weights
        )
    
    
    #model.backbone.init_weights(pretrained=pretrained_ckpt, bootstrap_method="centering")

    # --- Trainer ---
    visible_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))  # Respect SLURM allocation
    multi_gpu = visible_gpus > 1
    
    trainer = pl.Trainer(
        accelerator="gpu" ,
        devices=1,
        strategy='ddp_find_unused_parameters_true',
        precision="bf16-mixed",
        max_epochs=max_epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
        logger=logger,
        default_root_dir="eeg_classification/checkpoints",
        callbacks=[ checkpoint_cb]
        
    )

    # --- Train ---
    trainer.fit(model, train_loader, val_loader)

    # --- Test ---
    # --- Load Best Model ---
    best_model_path = checkpoint_cb.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = VisionTransformerEEG.load_from_checkpoint(best_model_path)
    #trainer.test(model, dataloaders=test_loader)

    # --- Collect Predictions ---
    y_true, y_pred, y_probs = [], [], []

    model.eval().cuda()
    
    with torch.no_grad():
        for x, y, age in val_loader:
            x, age = x.to(model.device), age.to(model.device)
            logits = model(x, age)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            """
            # ✨ Add this block once
            if len(y_true) == 0:
                x_vis = x[:1]
                attn = model.backbone.get_last_selfattention(x_vis)
                from model import visualize_attention_map
                visualize_attention_map(attn, model.backbone, save_path="eeg_classification/plots/attention_slice.png")
            """

    #plot_logs("eeg_classification/plots", trainer, y_true, y_pred)
    # --- t-SNE Visualisation ---
    print("Running t-SNE on validation CLS embeddings...")
    embeddings, labels = model.get_cls_embeddings(val_loader)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette='Set2')
    plt.title("t-SNE of CLS Token Embeddings (0=MCI, 1=DEM)")
    plt.savefig("eeg_classification/plots/tsne_cls_embedding.png")
    plt.show()

    plot_logs(
        trainer=trainer,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=np.array(y_probs)
    )

if __name__ == "__main__":
    main()
