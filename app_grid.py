import tkinter as tk
import numpy as np
from knn import MyKNN

# Размер сетки строго 28 на 28
GRID_SIZE = 28
CELL_SIZE = 14  # Размер одной клетки на экране (28 * 14 = 392 пикселя)


class PixelCanvasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Честный k-NN: Сетка 28x28")
        self.root.resizable(False, False)

        # 1. Загружаем данные и обучаем твой k-NN
        print("Загружаем базу и инициализируем MyKNN...")
        # Можно использовать mnist_full.npz или mnist_small.npz
        data = np.load("mnist_full.npz")
        self.model = MyKNN(k=3)
        self.model.fit(data["X_train"], data["y_train"])
        print("Готово к работе!")

        # 2. Наша честная матрица 28x28 (числа от 0.0 до 1.0)
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # 3. Холст
        canvas_px = GRID_SIZE * CELL_SIZE
        self.canvas = tk.Canvas(root, width=canvas_px, height=canvas_px, bg="black")
        self.canvas.pack(pady=10)

        # Рисуем тонкую сетку (подсказка для глаз)
        for i in range(GRID_SIZE):
            pos = i * CELL_SIZE
            self.canvas.create_line(pos, 0, pos, canvas_px, fill="#1a1a1a")
            self.canvas.create_line(0, pos, canvas_px, pos, fill="#1a1a1a")

        # Привязываем рисование мышкой
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # 4. Кнопки
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn_clear = tk.Button(btn_frame, text="Очистить", font=("Arial", 12), command=self.clear)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_predict = tk.Button(btn_frame, text="Распознать", font=("Arial", 12, "bold"), command=self.predict)
        self.btn_predict.pack(side=tk.LEFT, padx=10)

        self.label_result = tk.Label(root, text="Нарисуйте цифру по центру", font=("Arial", 16, "bold"), fg="blue")
        self.label_result.pack(pady=10)

    def set_pixel(self, r, c, intensity):
        """Красит клетку в сетке и на экране"""
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            # Не делаем пиксель темнее, чем он уже был
            if intensity > self.grid[r, c]:
                self.grid[r, c] = intensity

                # Переводим яркость (0.0 .. 1.0) в цвет hex (#000000 .. #ffffff)
                val = int(intensity * 255)
                color = f"#{val:02x}{val:02x}{val:02x}"

                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#1a1a1a")

    def paint(self, event):
        # Находим, в какую клетку 28x28 попал курсор
        c = event.x // CELL_SIZE
        r = event.y // CELL_SIZE

        # Кисть как в MNIST: центр яркий, вокруг полутень
        self.set_pixel(r, c, 1.0)
        self.set_pixel(r - 1, c, 0.6)
        self.set_pixel(r + 1, c, 0.6)
        self.set_pixel(r, c - 1, 0.6)
        self.set_pixel(r, c + 1, 0.6)

    def clear(self):
        self.grid.fill(0.0)
        self.canvas.delete("all")
        canvas_px = GRID_SIZE * CELL_SIZE
        # Восстанавливаем линии сетки
        for i in range(GRID_SIZE):
            pos = i * CELL_SIZE
            self.canvas.create_line(pos, 0, pos, canvas_px, fill="#1a1a1a")
            self.canvas.create_line(0, pos, canvas_px, pos, fill="#1a1a1a")
        self.label_result.config(text="Нарисуйте цифру по центру", fg="blue")

    def predict(self):
        # Проверяем, нарисовано ли хоть что-то
        if np.max(self.grid) == 0:
            self.label_result.config(text="Сетка пуста!", fg="red")
            return

        # Разворачиваем честные 28x28 в вектор длины 784
        x_new = self.grid.reshape(784)

        # Запускаем твой алгоритм!
        digit = self.model.predict_one(x_new)
        self.label_result.config(text=f"Результат k-NN: {digit}", fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelCanvasApp(root)
    root.mainloop()