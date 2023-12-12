import numpy as np

# Определение функции активации (сигмоидной функции)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Определение класса нейронной сети
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов со случайными значениями
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        # Прямое распространение
        self.z = np.dot(X, self.W1)
        self.z2 = sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        y_hat = sigmoid(self.z3)
        return y_hat

# Создание экземпляра нейронной сети
input_size = 2
hidden_size = 10
output_size = 2
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Входные данные
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Ожидаемые выходные данные
y = np.array([[0], [1], [1], [0]])

# Обучение нейронной сети
epochs = 10000
learning_rate = 0.1
for i in range(epochs):
    # Прямое распространение
    y_hat = nn.forward(X)

    # Обратное распространение (обновление весов)
    error = y - y_hat
    delta = error * sigmoid(y_hat) * (1 - sigmoid(y_hat))
    delta_hidden = np.dot(delta, nn.W2.T) * sigmoid(nn.z2) * (1 - sigmoid(nn.z2))
    nn.W2 += learning_rate * np.dot(nn.z2.T, delta)
    nn.W1 += learning_rate * np.dot(X.T, delta_hidden)

# Пример использования нейронной сети
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = nn.forward(test_input)
print("Predicted Output:")
print(predicted_output)