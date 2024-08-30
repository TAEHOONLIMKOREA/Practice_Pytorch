import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

#  맨 처음 한번만 다운로드 하기
dataset = MNIST('./data', transform=img_transform, download=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x



# # 기본 텐서 생성
# x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
# print(x)

# # 난수 텐서 생성
# x = torch.rand((2, 3))
# print(x)

# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])

# # 더하기
# c = a + b
# print(c)

# # 곱하기
# c = a * b
# print(c)

# GPU가 사용 가능한지 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("cpu")

# 텐서를 GPU로 이동
x = torch.tensor([1, 2, 3], device=device)

model = SimpleNN()
print(model)

criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):  # 100 에포크 동안 훈련
    # 예시 입력 및 출력 데이터
    inputs = torch.rand((32, 10))  # 배치 크기 32, 입력 크기 10
    targets = torch.rand((32, 1))  # 출력 크기 1

    # 모델 예측
    outputs = model(inputs)
    
    # 손실 계산
    loss = criterion(outputs, targets)
    
    # 옵티마이저 초기화
    optimizer.zero_grad()
    
    # 역전파
    loss.backward()
    
    # 가중치 업데이트
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
torch.save(model.state_dict(), 'model.pth')