import urllib.request
import numpy as np

print("Скачиваем и подготавливаем ВСЮ базу MNIST (60 000 эталонов)...")
url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
urllib.request.urlretrieve(url, "mnist_raw.npz")

with np.load("mnist_raw.npz") as data:
    # Забираем ВСЕ 60 000 картинок!
    X_train = data["x_train"].reshape(-1, 784) / 255.0
    y_train = data["y_train"]

    X_test = data["x_test"].reshape(-1, 784) / 255.0
    y_test = data["y_test"]

np.savez_compressed("mnist_full.npz",
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test)

print(f"Готово! Сохранена полная база: {X_train.shape[0]} картинок.")