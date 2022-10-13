import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 

device = 'cuda' 
#if torch.cuda.is_available()
#print('CUDA:', torch.cuda.is_available(), '     Use << {} >>'.format(device.upper()))
#print('PyTorch Version:', torch.__version__)   

# torchvision의 dataset을 이용하여 FashionMNIST를 가져오는 코드
# root은 파일을 저장할 경로를 만들어 준다
# train은 True일 시 training 데이터로 사용되고, False일 시 test 데이터로 사용된다
# transform은 다운받은 데이터의 format을 pytorch에서 사용하는 tensor로 변환해준다
train_data = datasets.FashionMNIST(    
    root='data', train=True, download=True, transform=ToTensor()
)
test_data = datasets.FashionMNIST(    
    root='data', train=False, download=True, transform=ToTensor()
)

# batch_size는 한번에 얼마나 train 할지를 정하는 부분

batch_size = 64

# dataloader는 dataset의 data를 반복 가능하게 해줌

trainloader = DataLoader(    
    train_data, batch_size=batch_size
)
testloader = DataLoader(    
    test_data, batch_size=batch_size
)

for X, y in testloader:
    print('Shape of X [N, C, H, W]:\n', X.shape)    
    print('Shape of y:\n', y.shape, '\n', y.dtype)    
    break

labels_map = {    
    0: "T-Shirt",    
    1: "Trouser",    
    2: "mantoman",    
    3: "Dress",    
    4: "Coat",    
    5: "Sandal",    
    6: "Shirt",    
    7: "Sneaker",    
    8: "Bag",    
    9: "Ankle Boot",
}

# flatten는 메소드는 2차원 데이터를 1차원 데이터로 바꿈
# layer는 nn.Squential로 정의해 network architecture을 정한다

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        Net_Out = self.layer(x)
        return Net_Out

# optimizer Traindataset의 실제 결과와 모델이 예측한 결과를 기반으로 최적화하는 도구

def train(dataloader, model, loss_fn, optimizer):
    pbar = tqdm(dataloader, desc=f'Training')
    for batch, (x, y) in enumerate(pbar):
        # CUDA를 이용하여 pytorch를 돌리는 경우 연산속도가 빨라지고
        # GPU에 tensor 옮겨줌
        X, y = X.to(device), y.to(device)
        
        # Frerdforward
        pred = model(x)

        # Calc, Loss
        loss = loss_fn(pred. y)

        # Backptopagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):

    # len을 사용하면 데이터의 갯수가 나오는 것을 이용해 사이즈를 구함
    # custom dataset일 때는 __len__ method를 새로 정의해서 data의 개수를 구해야함

    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    
    # model.eval은 
    # training 때와 inference 때 하는 역할이 다른 설정들을 적절하게 설정해준다

    model.eval()
    loss, correct = 0, 0

    # torch.no_grad는 지금 inference를 하니 
    # Neural Network의 patameter들에 default로 설정되있던
    # gradient tracking을 하지마 라는 의미로 쓰임

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # loss_fn의 반환값이 tensor이므로, item을 이용해 scalar 값 구함

            loss += loss_fn(pred, y).item()

            # 만든 Neural Network를 이용해 예측한 결과가 맞았는지 틀렸는지
            # 여부를 계산하고 그것을 토대로 accuracy를 구하는 부분
            # classification 결과를 체크하는 pred.argmax(1) == y
            # pred의 output은 banch_size * num_label의 결과이다
            # 보통 numpy나 torch로 argmax를 돌리면 input array를 넣어주고
            # argument로 axis를 넣는데
            # 이 코드는 tensor에 바로 argmax를 꽂는 방식이므로
            # torch.Tensor.argmax 문법이 됨

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batchs
    correct /= size
    print(f'Test Accuracy: {(100*correct):>0.1f}%    Loss: {loss:>8f}\n')

    return 100*correct, loss

# 위 코드에서 정의한 train/test function들을 사용하는 부분
# model은 Net class를 가지고 새롭게 정의한 variable에 해당한다

model = Net().to(device) 
    
# lr은 learning rate이다
# 계산된 gradient 값을 얼마나 크게 update할 것인지를 정하는 변수

lr = 1e-3
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr) 

fig = plt.figure(figsize=(20,5))
line1, line2 = plt.plot([],[],[],[])
plt.clf() 

epochs = 100
for t in range(epochs):
    print(f'----- Epoch {t+1} -----')
    train(trainloader, model, loss_fn, optimizer)
    accuracy, loss = test(testloader, model, loss_fn)
    # Add Accuracy & Loss to the Lines
    line1.set_xdata(np.append(line1.get_xdata(), t+1))
    line1.set_ydata(np.append(line1.get_ydata(), loss))
    line2.set_ydata(np.append(line2.get_ydata(), accuracy))

fig.add_subplot(1,2,1)
plt.plot(line1.get_xdata(), line1.get_ydata(), color='red')
plt.plot(line1.get_xdata(), line1.get_ydata(), 'o', color='red')
plt.xlabel('Epoch', fontsize=12); 
plt.ylabel('Loss', fontsize=12)
fig.add_subplot(1,2,2)
plt.plot(line1.get_xdata(), line2.get_ydata(), color='blue')
plt.plot(line1.get_xdata(), line2.get_ydata(), 'o', color='blue')
plt.xlabel('Epoch', fontsize=12); 
plt.ylabel('Accuracy', fontsize=12) 
plt.tight_layout()
plt.autoscale()
plt.show()


# softmax
# crossentropyloss
# network architecture
# optim