import torch as th
import numpy as np
import model as rat
import matplotlib.pyplot as plt

device = th.device('cuda:0')
model = th.load('save_Adam2.pt').to(device)
cost_rate = 0.0025
norisk_rate = 0.03


def get_loss(reward, ps, ps_last):
    value = th.abs(ps - ps_last)
    transaction_cost = value.sum() * cost_rate
    return (transaction_cost - th.dot(ps, reward)) * 100


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
price = th.cat([price1, price2, price3, price4, price5, price6], 0).transpose(0, 1).to(device)
reward = th.cat([reward1, reward2, reward3, reward4, reward5, reward6], 0).transpose(0, 1).to(device)
cash_p = th.ones(1, price.size(1), price.size(2)).to(device)
cash_r = th.ones(1, reward.size(1)).to(device)
price = th.cat([cash_p, price], 0)
reward = th.cat([cash_r, reward], 0)
model.eval()
asset = th.ones(1).to(device)
max_price = th.zeros(1).to(device)
min_price = th.ones(1).to(device) * 100
last_ps = th.randn(price.size(0)).to(device)
last_ps = th.nn.functional.softmax(last_ps)
print(last_ps)
x = np.arange(1, 91, 1)
list = []
for i in range(240, 330):
    list1 = []
    for j in range(30):
        list1.append((price[:, i + j, :] / price[:, i + 29, :]).detach())

    p = th.tensor([item.cpu().detach().numpy() for item in list1]).cuda().transpose(0, 1)
    ps = model(p, p[:, 25:, :], last_ps)
    print("ps:", ps)
    print("reward:", reward)
    asset *= -get_loss(reward[:, i + 29], ps, last_ps) / 100
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
