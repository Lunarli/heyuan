from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_true = np.array([0.28089125, 0.28089125, 0.27246592, 0.2696508 , 0.27528101, 0.27388343,
 0.27667858, 0.27246592, 0.2696508 , 0.26685568, 0.26685568, 0.27388343,
 0.2696508 , 0.262643  , 0.2598279 , 0.26543814, 0.26543814, 0.2696508 ,
 0.28510391, 0.29915947, 0.30195459, 0.28652145, 0.27667858, 0.27246592])

y_pred = np.array([0.28796693, 0.28294035, 0.2804649 , 0.28000487, 0.2777075 , 0.27846025,
 0.27888872, 0.28039878, 0.2764591 , 0.27780912, 0.27351992, 0.28212112,
 0.28246766, 0.29419252, 0.29677538, 0.29923895, 0.2832093 , 0.28339091,
 0.29145126, 0.2938422 , 0.30619709, 0.28956368, 0.2910939 , 0.29153747])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")


import numpy as np

# Given predicted and true values
true_values = np.array([0.28089125, 0.28089125, 0.27246592, 0.2696508, 0.27528101, 0.27388343, 0.27667858, 0.27246592, 0.2696508, 0.26685568, 0.26685568, 0.27388343, 0.2696508, 0.262643, 0.2598279, 0.26543814, 0.26543814, 0.2696508, 0.28510391, 0.29915947, 0.30195459, 0.28652145, 0.27667858, 0.27246592])
predicted_values = np.array([0.28796693, 0.28294035, 0.2804649, 0.28000487, 0.2777075, 0.27846025, 0.27888872, 0.28039878, 0.2764591, 0.27780912, 0.27351992, 0.28212112, 0.28246766, 0.29419252, 0.29677538, 0.29923895, 0.2832093, 0.28339091, 0.29145126, 0.2938422, 0.30619709, 0.28956368, 0.2910939, 0.29153747])

# Calculate the differences between predicted and true values
differences = predicted_values - true_values

# Calculate the squared differences
squared_differences = differences ** 2

# Calculate MSE
mse = np.mean(squared_differences)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate R2
mean_true_values = np.mean(true_values)
ss_total = np.sum((true_values - mean_true_values) ** 2)
ss_residual = np.sum(squared_differences)
r2 = 1 - (ss_residual / ss_total)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


from sklearn import tree
import matplotlib.pyplot as plt

# 创建决策树模型
clf = tree.DecisionTreeRegressor()

# 使用训练数据拟合决策树模型
clf.fit(X_train, y_train)

# 可视化决策树
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(clf, ax=ax)
plt.show()

