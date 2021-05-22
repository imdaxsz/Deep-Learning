import matplotlib.pyplot as plt

X = [0, 1, 2]
Y = [1, 3, 2]
#yhat = a x + b

a = 0.1
b = 1.0
LR = 0.1
Loss = []

for k in range(50):
    delta_a = 0
    delta_b = 0
    E = 0
    for i in range(3):
        yHat = a * X[i] + b
        delta_a += LR * (Y[i] - yHat) * X[i]
        delta_b += LR * (Y[i] - yHat) * 1
        E += (Y[i] - yHat)**2
    E /= 2
    Loss.append(E)
    delta_a /= 3 # 델타값 평균 적용
    delta_b /= 3
    a = a + delta_a
    b = b + delta_b
    print('{0} : a={1:0.2f} b={2:0.2f}, loss={3:0.8f}'.format(k, a, b, E))

#plt.figure(figsize=(10, 4))
plt.plot(Loss)
plt.show()


