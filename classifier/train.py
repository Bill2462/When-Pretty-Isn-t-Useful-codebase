import os
import math
import torch
import argparse

import torch.nn.functional as F
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from dataset import (ImageDataset,
                     rgb_space_transform,
                     bp_filtered_rgb_space_transform,
                     depth_space_transform,
                     texture_transform)

from utils import AverageMeter, append_to_logfile

def cosine_annealing_with_linear_warmup(n_steps, n_steps_max, n_warmup_steps, start_lr, min_lr):
    if n_steps < n_warmup_steps:
        return float(n_steps) / float(max(1, n_warmup_steps)) * (1 - start_lr) + start_lr
    progress = float(n_steps - n_warmup_steps) / float(max(1, n_steps_max - n_warmup_steps))
    return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

def get_args():
    parser = argparse.ArgumentParser(description='Train a feature extractor.')
    
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Root directory containing training data')
    
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='Root directory containing validation data')
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model and logs')
    
    parser.add_argument('--model_arch', type=str, choices=['resnet50', 'resnet18',
                                                           'vit_tiny_patch16_224',
                                                           'convnext_tiny', 'swin_v2_tiny',
                                                           'bagnet9', 'bagnet17', 'bagnet33'],
                        default='resnet50',
                        help='Model architecture to use for training')

    parser.add_argument('--data_type', type=str, choices=['rgb', 'rgb_bp', 'depth', 'texture'],
                        default='rgb',
                        help='Type of data to use for training')
    
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    
    parser.add_argument('--n_epochs', type=int, default=80,
                        help='Number of epochs for training')
    
    parser.add_argument('--n_warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs with linear learning rate increase')
    
    parser.add_argument('--early_stopping_patience', type=int, default=100_000,
                        help='Number of evaluations with no improvement to wait before stopping training')
    
    parser.add_argument('--log_every_n_steps', type=int, default=100,
                        help='Log training metrics every n steps')
    
    parser.add_argument('--minimum_lr', type=float, default=1e-8,
                        help='Minimum learning rate for the scheduler')
    
    parser.add_argument('--start_lr', type=float, default=1e-5,
                        help='Starting learning rate for the scheduler')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    parser.add_argument('--num_classes', type=int, default=200,
                        help='Number of classes for classification')
    
    parser.add_argument('--num_workers', type=int, default=24,
                        help='Number of data loading workers')
    
    parser.add_argument('--gradient_clip_threshold', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the last checkpoint.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda or cpu)')

    return parser.parse_args()

class TrainingState:
    def __init__(self,
                 model,
                 optimizer,
                 scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_epoch = 0
        self.current_step = 0
        self.early_stopping_counter = 0
        self.best_val_score = float('inf')
    
    def save(self, path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_score': self.best_val_score,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'early_stopping_counter': self.early_stopping_counter
        }
        
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(state, path)
    
    def load(self, path):
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        self.best_val_score = state['best_val_score']
        self.current_epoch = state['current_epoch']
        self.current_step = state['current_step']
        self.early_stopping_counter = state['early_stopping_counter']

    def resume(self, output_dir):
        checkpoint_path = os.path.join(output_dir, "current.ckpt")
        self.load(checkpoint_path)

def build_lr_scheduler(optimizer,
                       n_epochs: int,
                       n_warmup_epochs: int,
                       start_lr: float,
                       min_lr: float,
                       dataset_size: int,
                       batch_size: int):
    n_steps = (dataset_size // batch_size) * n_epochs
    n_warmup_steps = (dataset_size // batch_size) * n_warmup_epochs

    def lr_lambda(current_step):
        return cosine_annealing_with_linear_warmup(current_step,
                                                  n_steps,
                                                  n_warmup_steps,
                                                  start_lr,
                                                  min_lr)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@torch.no_grad()
def eval(model, dataloader, device):
    model.eval()

    loss_tracker = AverageMeter()
    acc_tracker = AverageMeter()

    for batch in dataloader:
        imgs, targets = batch['image'].to(device), batch['label'].to(device)
        
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean().item()

        acc_tracker.update(acc, imgs.size(0))
        loss_tracker.update(loss.detach().item(), imgs.size(0))

    return {
        "loss": loss_tracker.avg,
        "accuracy": acc_tracker.avg
    }

def train_epoch(training_state: TrainingState,
                train_loader: DataLoader,
                device: str,
                log_every_n_steps: int,
                log_filepath: str,
                gradient_clip_threshold: float,
                cutmix: v2.CutMix = None):
    training_state.model.train()
    train_loss_tracker = AverageMeter()
    train_loss_global = AverageMeter()
    for step, batch in enumerate(train_loader):
        labels, images = batch["label"], batch["image"]
        
        if cutmix is not None:
            images, labels = cutmix(images, labels)
        
        labels, images = labels.to(device), images.to(device)
        training_state.optimizer.zero_grad()

        outputs = training_state.model(images)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        
        if gradient_clip_threshold:
            torch.nn.utils.clip_grad_norm_(training_state.model.parameters(), gradient_clip_threshold)
        
        training_state.optimizer.step()

        if training_state.scheduler:
            training_state.scheduler.step()

        train_loss_tracker.update(loss.detach().item(), images.size(0))
        train_loss_global.update(loss.detach().item(), images.size(0))
        lr =  training_state.optimizer.param_groups[0]['lr']
        if (step + 1) % log_every_n_steps == 0 and (step + 1) > 0:
            append_to_logfile(log_filepath, {
                "type": "train",
                "step": training_state.current_step + 1,
                "lr": lr,
                "loss": train_loss_tracker.avg,
            })

            train_loss_tracker.reset()
        
        training_state.current_step += 1

    return train_loss_global.avg

def run_training(training_state: TrainingState,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str,
                 n_epochs: int,
                 gradient_clip_threshold: float,
                 log_every_n_steps: int,
                 early_stopping_patience: int,
                 logdir: str,
                 cutmix: v2.CutMix = None):
    log_filepath = os.path.join(logdir, 'log.json')
    training_state.model.train()

    for epoch in range(training_state.current_epoch, n_epochs):
        train_loss = train_epoch(training_state,
                                 train_loader,
                                 device,
                                 log_every_n_steps,
                                 log_filepath,
                                 gradient_clip_threshold,
                                 cutmix=cutmix)
        
        val_metrics = eval(training_state.model, val_loader, device)
        lr = training_state.optimizer.param_groups[0]['lr']

        append_to_logfile(log_filepath, {
            "type": "val",
            "epoch": epoch,
            "lr": lr,
            "step": training_state.current_step,
            "loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"]
        })

        print(f"Epoch {epoch}: "
              f"lr: {lr:.6f}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_metrics['loss']:.4f}, "
              f"Validation Accuracy: {val_metrics['accuracy']:.4f}", flush=True)
        
        if val_metrics["loss"] < training_state.best_val_score:
            training_state.best_val_score = val_metrics["loss"]
            training_state.save(os.path.join(logdir, 'best.ckpt'))
        
            training_state.early_stopping_counter = 0
        else:
            training_state.early_stopping_counter += 1
            if training_state.early_stopping_counter >= early_stopping_patience:
                print(f"No improvement in validation loss for {early_stopping_patience} evaluations. Stopping training.", flush=True)
                break
        
        training_state.save(os.path.join(logdir, "current.ckpt"))
        training_state.current_epoch += 1

def prepare_dl(train_data_path: str,
               val_data_path: str,
               data_type: str,
               batch_size: int,
               num_workers: int,
               prefetch_factor: int = 3):
    if data_type == 'rgb':
        train_transform = rgb_space_transform(is_test=False, imsize=224)
        val_transform = rgb_space_transform(is_test=True, imsize=224)
    elif data_type == 'depth':
        train_transform = depth_space_transform(is_test=False, imsize=224)
        val_transform = depth_space_transform(is_test=True, imsize=224)
    elif data_type == 'texture':
        train_transform = texture_transform(is_test=False, resize_size=256, crop_size=128)
        val_transform = texture_transform(is_test=True, resize_size=256, crop_size=128)
    elif data_type == 'rgb_bp':
        train_transform = bp_filtered_rgb_space_transform(is_test=False, imsize=224)
        val_transform = bp_filtered_rgb_space_transform(is_test=True, imsize=224)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    train_dataset = ImageDataset(train_data_path, transform=train_transform, npy_mode=(data_type=='rgb_bp'))
    val_dataset = ImageDataset(val_data_path, transform=val_transform, npy_mode=(data_type=='rgb_bp'))

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                persistent_workers=True,
                                prefetch_factor=prefetch_factor,
                                pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                drop_last=False)
    
    return train_loader, val_loader, len(train_dataset)

def get_model_architecture(model_arch: str, num_classes: int):
    if model_arch == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(num_classes=num_classes)
    
    elif model_arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(num_classes=num_classes)
    
    elif model_arch == 'bagnet9':
        from bag_net import bagnet9
        model = bagnet9(num_classes=num_classes)
    
    elif model_arch == 'bagnet17':
        from bag_net import bagnet17
        model = bagnet17(num_classes=num_classes)
    
    elif model_arch == 'bagnet33':
        from bag_net import bagnet33
        model = bagnet33(num_classes=num_classes)
    
    elif model_arch == 'vit_tiny_patch16_224':
        from timm import create_model
        model = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    
    elif model_arch == 'convnext_tiny':
        from torchvision.models import convnext_tiny
        model = convnext_tiny(num_classes=num_classes)
    
    elif model_arch == 'swin_v2_tiny':
        from torchvision.models import swin_t
        model = swin_t(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")
    
    return model

def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, train_dataset_size = prepare_dl(
        args.train_data_path,
        args.val_data_path,
        args.data_type,
        args.batch_size,
        args.num_workers,
    )

    model = get_model_architecture(args.model_arch, args.num_classes)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    cutmix = v2.CutMix(num_classes=args.num_classes)
    
    scheduler = build_lr_scheduler(optimizer,
                                   n_epochs=args.n_epochs,
                                   n_warmup_epochs=args.n_warmup_epochs,
                                   start_lr=args.start_lr,
                                   min_lr=args.minimum_lr,
                                   dataset_size=train_dataset_size,
                                   batch_size=args.batch_size)

    state = TrainingState(model=model,
                          optimizer=optimizer,
                          scheduler=scheduler)
    
    if args.resume:
        state.resume(args.output_dir)

    run_training(state,
                 train_loader,
                 val_loader,
                 device=args.device,
                 n_epochs=args.n_epochs,
                 gradient_clip_threshold=args.gradient_clip_threshold,
                 log_every_n_steps=args.log_every_n_steps,
                 early_stopping_patience=args.early_stopping_patience,
                 logdir=args.output_dir,
                 cutmix=cutmix)

if __name__ == "__main__":
    main()
