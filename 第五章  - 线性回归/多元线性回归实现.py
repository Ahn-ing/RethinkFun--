# Feature 数据
X = [[10, 3], [20, 3], [25, 3], [28, 2.5], [30, 2], [35, 2.5], [40, 2.5]]
# Label 数据
y = [60, 85, 100, 120, 140, 145, 163]


class MLR:
    def __init__(self, x, y, lr=0.0001):
        self.x = x
        self.y = y
        self.n = len(y)
        self.lr = lr
        self.weights = [0.0 for _ in range(len(x[0]) + 1)]

    def output(self):
        y_pred = []
        for i in range(self.n):
            y_hat = (
                self.weights[0]
                + self.weights[1] * self.x[i][0]
                + self.weights[2] * self.x[i][1]
            )
            y_pred.append(y_hat)
        return y_pred

    def loss(self, y_pred):
        total_loss = 0.0
        for i in range(self.n):
            total_loss += (y_pred[i] - self.y[i]) ** 2
        return total_loss / self.n

    def update(self, y_pred):
        grad_w0 = 2 * sum(y_pred[i] - self.y[i] for i in range(self.n)) / self.n
        grad_w1 = (
            2
            * sum((y_pred[i] - self.y[i]) * self.x[i][0] for i in range(self.n))
            / self.n
        )
        grad_w2 = (
            2
            * sum((y_pred[i] - self.y[i]) * self.x[i][1] for i in range(self.n))
            / self.n
        )
        self.weights[0] -= self.lr * grad_w0
        self.weights[1] -= self.lr * grad_w1
        self.weights[2] -= self.lr * grad_w2

if __name__ == "__main__":
    model = MLR(X, y)
    for epoch in range(10000):
        y_pred = model.output()
        loss = model.loss(y_pred)
        model.update(y_pred)
        if epoch % 100 == 0:
            print(f"epoch {epoch:.5f}:loss {loss:.5f}")

    print(
        f"最终的参数为w0={model.weights[0]:.5f},w1={model.weights[1]:.5f},w2={model.weights[2]:.5f}"
    )
