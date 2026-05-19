import torch
import torchvision.models as models
import torch.nn as nn

# Load ResNet50 - training mode
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
model.fc = nn.Identity()
model.train()  # training mode (enables dropout, batchnorm tracking)

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Tạo 1 cái input 2048 x 2048 - requires_grad for backprop
x = torch.randn(1, 1, 1536, 1536, requires_grad=True, device=device)

# Forward qua CNN (NO torch.no_grad() - cần giữ computation graph)
output = model(x)

print("Input shape: ", x.shape)
print("Output shape:", output.shape)
allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
max_allocated = torch.cuda.max_memory_allocated() / 1024**2

print("-------------------------------------------------------")
print(f"GPU allocated: {allocated:.2f} MB")
print(f"GPU reserved: {reserved:.2f} MB")
print(f"Max allocated: {max_allocated:.2f} MB")
print("-------------------------------------------------------")
# --- Backprop example ---
criterion = torch.nn.CrossEntropyLoss()
target = torch.zeros(1, dtype=torch.long, device=device)  # fake label class 0

loss = criterion(output, target)
print(f"Loss: {loss.item()}")

loss.backward()  # backprop

print("Gradient on input:", x.grad.shape)  # confirms graph was kept
print("Gradient on first conv weight:", model.conv1.weight.grad.shape)