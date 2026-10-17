import numpy as np


class MyKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def distance(self, a, b):
        total = 0.0
        for i in range(len(a)):
            diff = a[i] - b[i]
            total += diff ** 2
        return total ** 0.5

    def count(self, c):
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        max_value = 0
        max_key = 0

        # 1. Считаем голоса (коротко и элегантно)
        for i in c:
            counts[i] += 1

        # 2. Ищем победителя
        for key, value in counts.items():
            if value > max_value:
                max_value = value
                max_key = key

        return max_key

    def predict_one(self, x_new):
        # 1. Считаем расстояния до ВСЕХ 60 000 картинок одной строкой без цикла for!
        # diff — это матрица (60000, 784)
        diff = self.X_train - x_new

        # distances — массив из 60000 чисел (сумма квадратов по каждой строке)
        distances = np.sum(diff ** 2, axis=1)

        # 2. Находим k ближайших (все как у тебя было)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:self.k]

        # 3. Достаем цифры
        neighbors = [self.y_train[i] for i in k_nearest_indices]

        # 4. Твой подсчет голосов
        return self.count(neighbors)
if __name__ == "__main__":
    print("Загружаем данные...")
    data = np.load("mnist_small.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("Обучаем модель...")
    model = MyKNN(k=7)
    model.fit(X_train, y_train)

    print("Проверяем на тестовой картинке №0...")
    true_label = y_test[0]
    prediction = model.predict_one(X_test[0])

    print("--------------------------------")
    print(f"Правильный ответ: {true_label}")
    print(f"Твой k-NN ответил: {prediction}")
    print("--------------------------------")
