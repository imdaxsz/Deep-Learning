import matplotlib.pyplot as plt

X = [0, 1, 2]
Y = [1, 3, 2]

#yhat = a x + b
def Test(a, b, lr):
    Loss = []
    for k in range(50):
        delta_a = 0
        delta_b = 0
        E = 0
        for i in range(3):
            yHat = a * X[i] + b
            delta_a += lr * (Y[i] - yHat) * X[i]
            delta_b += lr * (Y[i] - yHat) * 1
            E += (Y[i] - yHat)**2
        E /= 2
        Loss.append(E)
        delta_a /= 3
        delta_b /= 3
        a = a + delta_a
        b = b + delta_b
        print('{0} : a={1:0.2f} b={2:0.2f}, loss={3:0.8f}'.format(k, a, b, E))
    return Loss

figure, axes = plt.subplots(3, 3, figsize=(10, 6))
plt.tight_layout()

a = 0.1
b = 0.0
lr = 0.0

for i in range(3):
    for j in range(3):
        lr += 0.1
        loss = Test(a, b, lr)
        axes[i][j].set_title('LR={:0.1}'.format(lr))
        #axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])
        axes[i][j].plot(loss)
plt.show()


