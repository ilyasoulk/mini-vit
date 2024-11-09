from model import ViT
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # training on mnist
    patch_size = 4 # 4x4
    input_shape = (1, 28, 28)
    hidden_dim = 256
    fc_dim = 512
    heads = 4
    activation = "gelu"
    num_classes = 10
    layers = 2

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
    vit_small = ViT(
        patch_size=patch_size,
        input_shape=input_shape,
        hidden_dim=hidden_dim,
        fc_dim=fc_dim,
        num_heads=heads,
        activation=activation,
        num_classes=num_classes,
        num_blocks=layers
    )

    num_param = sum(param.numel() for param in vit_small.parameters())
    print(f"number of parameters : {num_param}")

    criterion =  nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit_small.parameters(), lr=0.001)

    epochs = 4
    device = torch.device("mps")

    vit_small = vit_small.to(device)

    for epoch in range(epochs):
        vit_small.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            data = data.to(device)
            target = target.to(device)
            output = vit_small(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')




    vit_small.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = vit_small(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
