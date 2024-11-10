import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import ViT
from typing import Tuple, Dict
# import wandb  # Optional, for logging

def get_model_config() -> Dict:
    return {
        'patch_size': 4,
        'input_shape': (3, 32, 32),
        'hidden_dim': 384,
        'fc_dim': 1536,
        'num_heads': 6,
        'num_blocks': 7,
        'num_classes': 10,
        'activation': 'gelu'
    }

def get_train_config() -> Dict:
    return {
        'batch_size': 128,
        'epochs': 100,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'device': torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    }

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    return transform_train, transform_test

def get_dataloaders(transform_train, transform_test, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform_train, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform_test, download=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def train_one_epoch(model: nn.Module, 
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device,
                   epoch: int) -> float:
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f} Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader)

def evaluate(model: nn.Module, 
            test_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%')
    
    return test_loss, accuracy

def main():
    # Get configurations
    model_config = get_model_config()
    train_config = get_train_config()
    
    # Setup device
    device = train_config['device']
    print(f"Using device: {device}")
    
    # Get data transforms and loaders
    transform_train, transform_test = get_transforms()
    train_loader, test_loader = get_dataloaders(
        transform_train, transform_test, train_config['batch_size']
    )
    
    # Initialize model
    model = ViT(**model_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=train_config['epochs']
    )
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(train_config['epochs']):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, 'best_model.pth')
        
        # Optional: Log metrics with wandb
        # wandb.log({
        #     'train_loss': train_loss,
        #     'test_loss': test_loss,
        #     'accuracy': accuracy,
        #     'learning_rate': scheduler.get_last_lr()[0]
        # })
    
    print(f'Best accuracy: {best_accuracy:.2f}%')

if __name__ == "__main__":
    # Optional: Initialize wandb
    # wandb.init(
    #     project="vit-cifar10",
    #     config={**get_model_config(), **get_train_config()}
    # )
    
    main()
