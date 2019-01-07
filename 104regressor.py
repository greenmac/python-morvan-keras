# https://morvanzhou.github.io/tutorials/machine-learning/keras/2-1-regressor/
# https://github.com/MorvanZhou/tutorials/blob/master/kerasTUT/4-regressor_example.py
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160] # first 160 data points
X_test, Y_test = X[160:], Y[160:] # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential() # 是添加到模型上的层的list
model.add(Dense(units=1, input_dim=1))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# train
print('Training ----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test
print('\nTesting ----------')
cost = model.evaluate(X_test, Y_test, batch_size=40) # 用 model.evaluate 取得 loss 值。若在 compile 時有指定 metrics，這裡也會回傳 metrics。
print('test cost: ', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# plotting the prediction
Y_pred = model.predict(X_test) # model.predict 就属于 Model 的功能，用来训练模型，用训练好的模型预测
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()