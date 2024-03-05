import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import model as rat
import torch.optim as optim
import matplotlib.pyplot as plt


th.autograd.set_detect_anomaly(True)
device = th.device('cuda:0')

cost_rate = 0.0025
norisk_rate = 0.03
model = rat.RAT(30, 2, 1, 4, 4, 5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)


def generate_data():
    price1 = th.load('dataset/ashare_sse300/tensor_data/ohlc_series_context_1.pt')
    price2 = th.load('dataset/ashare_sse300/tensor_data/ohlc_series_context_2.pt')
    price3 = th.load('dataset/ashare_sse300/tensor_data/ohlc_series_context_3.pt')
    price4 = th.load('dataset/ashare_sse300/tensor_data/ohlc_series_context_4.pt')
    price5 = th.load('dataset/ashare_sse300/tensor_data/ohlc_series_context_5.pt')
    price6 = th.load('dataset/ashare_sse300/tensor_data/ohlc_series_context_6.pt')
    reward1 = th.load('dataset/ashare_sse300/tensor_data/reward_tensor_1.pt')
    reward2 = th.load('dataset/ashare_sse300/tensor_data/reward_tensor_2.pt')
    reward3 = th.load('dataset/ashare_sse300/tensor_data/reward_tensor_3.pt')
    reward4 = th.load('dataset/ashare_sse300/tensor_data/reward_tensor_4.pt')
    reward5 = th.load('dataset/ashare_sse300/tensor_data/reward_tensor_5.pt')
    reward6 = th.load('dataset/ashare_sse300/tensor_data/reward_tensor_6.pt')
    price = th.cat([price1, price2, price3, price4, price5, price6], 0).transpose(0, 1)
    reward = th.cat([reward1, reward2, reward3, reward4, reward5, reward6], 0).transpose(0, 1)
    cash_p = th.ones(1, price.size(1), price.size(2))
    cash_r = th.ones(1, reward.size(1))
    price = th.cat([cash_p, price], 0)
    reward = th.cat([cash_r, reward], 0)
    return price[:, :, :].to(device), reward[:, :].to(device)


def get_loss(reward, ps, ps_last):
    value = th.abs(ps - ps_last)
    transaction_cost = value.sum() * cost_rate
    return (transaction_cost - th.dot(ps, reward)) * 100


data_price, data_reward = generate_data()
model.train()
with th.autograd.set_detect_anomaly(True):
    for epoch in range(30):
        print("start step:", epoch+1)
        time_stamp = 0
        last_ps = F.softmax(th.randn(data_price.size(0))).to(device)
        for i in range(240):
            loss = th.ones(1, requires_grad=True).to(device)
            list = []
            for j in range(30):
                list.append((data_price[:, i+j, :] / data_price[:, i+29, :]).detach())
            p = th.tensor([item.cpu().detach().numpy() for item in list]).cuda().transpose(0, 1)
            ps = model(p, p[:, 25:, :], last_ps)
            print("ps:", ps)
            print("reward:", data_reward[:, i+29])
            loss = loss * -get_loss(data_reward[:, i + 29], ps, last_ps.detach())
            time_stamp += 1
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            last_ps = ps


model.eval()
asset = th.ones(1).to(device)
max_price = th.zeros(1).to(device)
min_price = th.ones(1).to(device) * 100
last_ps = th.ones(data_price.size(0)).to(device) / data_price.size(0)
x = np.arange(1, 91, 1)
list = []
for i in range(240, 330):
    list1 = []
    for j in range(30):
        list1.append((data_price[:, i + j, :] / data_price[:, i + 29, :]).detach())

    p = th.tensor([item.cpu().detach().numpy() for item in list1]).cuda().transpose(0, 1)
    ps = model(p, p[:, 25:, :], last_ps)
    asset *= -get_loss(data_reward[:, i + 29], ps, last_ps) / 100
    list.append(asset.cpu().detach().numpy())
    max_price = th.max(asset, max_price)
    min_price = th.min(asset, min_price)

year_asset = asset ** 4
net_return = th.tensor(list) - th.tensor(1.)
return_std = th.std(net_return)
volatility = th.sqrt(th.tensor(252)) * return_std
sharpe = (year_asset - 1 - norisk_rate) / volatility
print(year_asset, volatility, sharpe, max_price.data-min_price.data)
y = np.array(list)
plt.plot(x, y)
plt.show()
th.save(model, 'save_Adam2.pt')
