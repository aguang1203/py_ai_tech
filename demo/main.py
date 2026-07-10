import pandas as pd

data = pd.read_csv('/home/hjg/dev/project/py_ai_tech/datasets/house-val.csv')

x = data["area"]
y = data["price"]

import matplotlib.pyplot as plt

w = 0
b = 0

def predict(w,b):
    y_pred = w * x + b
    plt.plot(x,y_pred,color="blue",label="pred")
    plt.scatter(x,y,marker="x",color="red",label="true")
    plt.title("house price")
    plt.xlabel("Area")
    plt.ylabel("Price")
    
    plt.xlim(0,250)
    plt.ylim(0,1600)

    plt.legend()
    plt.savefig("./demo/house_pred.png") 

# predict(7,100)

def cost_func(w,b):
    y_pred = w * x + b
    cost = (y - y_pred) ** 2
    return cost.sum() / len(x)

# costs = []
# for w in range(-100,100):
#     costs.append(cost_function(w,0))

# print(costs)

# plt.plot(range(-100,100),costs)
# plt.title("cost function")
# plt.xlabel("w")
# plt.ylabel("cost")
# plt.savefig("./demo/cost_function.png")


# import numpy as np
# ws = np.arange(-100,100)
# bs = np.arange(-100,100)
# costs = np.zeros((len(ws),len(bs)))

# for i,w in enumerate(ws):
#     for j,b in enumerate(bs):
#         costs[i,j] = cost_function(w,b)

# ax = plt.axes(projection="3d")
# ax.xaxis.set_pane_color((1,1,1,1))
# ax.yaxis.set_pane_color((1,1,1,1))
# ax.zaxis.set_pane_color((1,1,1,1))

# b_grid, w_grid = np.meshgrid(bs,ws)

# ax.plot_surface(w_grid,b_grid,costs,cmap="viridis",alpha=0.7)
# ax.set_title("cost function")
# ax.set_xlabel("w")
# ax.set_ylabel("b")
# ax.set_zlabel("cost")

# w_index,b_index = np.where(costs == np.min(costs))

# ax.scatter(ws[w_index],bs[b_index],costs[w_index,b_index],color="red",s=100)

# plt.savefig("./demo/cost_function_3d.png")

def gradient_func(w,b):
    y_pred = w * x + b
    w_grad = (2 * x * (y_pred - y)).mean()
    b_grad = (2 * (y_pred - y)).mean()
    return w_grad, b_grad

w = 0
b = 0 

for i in range(10):
    learning_rate = 0.01
    w_grad, b_grad = gradient_func(w,b)
    w = w - learning_rate * w_grad
    b = b - learning_rate * b_grad

    print(w,b)
    print(cost_func(w,b))

