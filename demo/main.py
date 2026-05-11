import pandas as pd

data = pd.read_csv('/home/hjg/dev/project/py_ai_tech/datasets/house-val.csv')
print(data)

x = data["area"]
y = data["price"]

import matplotlib.pyplot as plt

plt.scatter(x, y,marker='x',color='red')
plt.title('House Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.savefig('./demo/simple_plot.png')
plt.show()