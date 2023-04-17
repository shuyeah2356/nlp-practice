import numpy as numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import d2l.torch as d2l

# 使用正弦函数和一些加性噪声来生成序列数据，时间步为1，2，...,1000
T = 1000
time = torch.arange(1, T+1, dtype=torch.float32)
x = torch.sin(0.01*time)+torch.normal(0, 0.2, (T,))

plt.plot(time.numpy(), x.numpy())
plt.savefig("./generate_time_series.png")

# 将序列转化为特征标签对，生成数据对的个数比原始序列的长度少tau个，因为没有足够的历史数据来描述前tau个数据样本
tau = 4
features = torch.zeros((T-tau, tau))     # 去掉数据中前tau个数据，生成一个维度为（T-tau, tau）

for i in range(tau):
    features[:, i] = x[i:T-tau+i]
    print(x[i:T-tau+i].size())

labels = x[tau:].reshape((-1, 1))


batch_size, n_train = 16, 600
# dataloader
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 构建训练模型
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(tau, 2*tau),
            nn.ReLU(),
            nn.Linear(2*tau, 1)
        )
        
    def forward(self,x):
        x = self.net(x)


def get_net():
    net = nn.Sequential(nn.Linear(tau, 2*tau),
                        nn.ReLU(),
                        nn.Linear(2*tau, 1))

    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(),lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f"epoch{epoch+1},"
              f"loss:{d2l.evaluate_loss(net, train_iter, loss):f}"
        )

# net = MyNetwork()
net = get_net()
train(net, train_iter, loss, 5, 0.01)


# 单步预测
one_step_pred = net(features)
# print(one_step_pred)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), one_step_pred.detach().numpy()], 
         "time", "x", 
         legend=["data","1_step_pred"],
         xlim=[1, 1000],
         figsize=(6, 3))
plt.savefig('one_step_pred.png')

# 多步预测
multi_step_pred = torch.zeros(T)

multi_step_pred[:n_train+tau] = x[:n_train+tau]
for i in range(n_train+tau, T):
    multi_step_pred[i] = net(multi_step_pred[i-tau:i].reshape((1, -1)))
d2l.plot([time, time[tau:], time[n_train+tau:]],
        [x.detach().numpy(),one_step_pred.detach().numpy(),
        multi_step_pred[n_train+tau:].detach().numpy()], "time","x", legend=["data", "1-step pred", "multi-step pred"],xlim=[1, 1000],figsize=(6, 3))
plt.savefig('multi-step pred.png')


# k步预测
max_step=64
features = torch.zeros((T-tau-max_step+1, tau+max_step))
for i in range(tau):
    features[:, i] = x[i: i+T-tau-max_step+1]

for i in range(tau, tau+max_step):
    features[:, i] = net(features[:, i-tau:i]).reshape(-1)
steps = (1, 4, 16, 64)
d2l.plot([time[tau+i-1:T-max_step+i] for i in steps],
         [features[:, tau+i-1].detach().numpy() for i in steps], "time", "x",
         legend = [f"{i}-step preds" for i in steps],xlim=[5, 1000],
         figsize=(6, 3)
        )
plt.savefig("k_step_pred.png")