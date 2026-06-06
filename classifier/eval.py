import os
import gzip
import json
import torch
import argparse

import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (ImageDataset,
                     rgb_space_transform,
                     depth_space_transform,
                     bp_filtered_rgb_space_transform,
                     texture_transform)

from utils import AverageMeter

def get_args():
    parser = argparse.ArgumentParser(description='Train a feature extractor.')

    parser.add_argument('--data_path', type=str, required=True,
                        help='Root directory containing eval data')
    
    parser.add_argument('--ckpt_filepath', type=str, required=True,
                        help='Path to the model checkpoint file for evaluation')
    
    parser.add_argument('--output_filepath', type=str, required=True,
                        help='Path to save evaluation results')
    
    parser.add_argument('--model_arch', type=str, choices=['resnet50', 'resnet18',
                                                           'vit_tiny_patch16_224',
                                                           'convnext_tiny', 'swin_v2_tiny',
                                                           'bagnet9', 'bagnet17', 'bagnet33'],
                        default='resnet50',
                        help='Model architecture to use for training')

    parser.add_argument('--data_type', type=str, choices=['rgb', 'depth', 'texture', 'rgb_bp'],
                        default='rgb',
                        help='Type of data to use for training')
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    
    parser.add_argument('--num_classes', type=int, default=200,
                        help='Number of classes for classification')
    
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

@torch.no_grad()
def eval(model, dataloader, device):
    model.eval()

    loss_tracker = AverageMeter()
    acc_tracker = AverageMeter()
    ground_truth_targets = []
    logits = []

    for batch in dataloader:
        imgs, targets = batch['image'].to(device), batch['label'].to(device)
        
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean().item()

        acc_tracker.update(acc, imgs.size(0))
        loss_tracker.update(loss.detach().item(), imgs.size(0))

        ground_truth_targets.append(targets.cpu().numpy().tolist())
        logits.append(outputs.cpu().numpy().tolist())

    return {
        "loss": loss_tracker.avg,
        "accuracy": acc_tracker.avg,
        "ground_truths": ground_truth_targets,
        "logits": logits,
    }

def prepare_dl(data_path: str,
               data_type: str,
               batch_size: int,
               num_workers: int):

    if data_type == 'rgb':
        transform = rgb_space_transform(is_test=True, imsize=224)
    elif data_type == 'depth':
        transform = depth_space_transform(is_test=True, imsize=224)
    elif data_type == 'texture':
        transform = texture_transform(is_test=True, resize_size=256, crop_size=128)
    elif data_type == 'rgb_bp':
        transform = bp_filtered_rgb_space_transform(is_test=True, imsize=224)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    ds = ImageDataset(data_path, transform=transform, npy_mode=(data_type=='rgb_bp'))
    
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False)
    
    return dl

def get_model(model_arch: str, num_classes: int):
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

    dl = prepare_dl(
        args.data_path,
        args.data_type,
        args.batch_size,
        args.num_workers,
    )

    model = get_model(args.model_arch, args.num_classes)
    checkpoint = torch.load(args.ckpt_filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(args.device)
    
    eval_metrics = eval(model, dl, args.device)

    os.makedirs(os.path.dirname(args.output_filepath), exist_ok=True)
    with gzip.open(args.output_filepath, 'wt', encoding='utf-8') as f:
        json.dump(eval_metrics, f, indent=4)

    print(f"Evaluation results saved to {args.output_filepath}")

if __name__ == "__main__":
    main()
