import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from knn import MyKNN


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознаватель цифр (k-NN)")
        self.root.resizable(False, False)

        # 1. Загружаем данные и обучаем твой k-NN
        print("Загружаем базу и инициализируем MyKNN...")
        data = np.load("mnist_full.npz")
        self.model = MyKNN(k=3)
        self.model.fit(data["X_train"], data["y_train"])
        print("Готово к работе!")

        # 2. Холст для рисования (280x280 пикселей)
        self.canvas_size = 280
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack(pady=10)

        # Создаем параллельно невидимую картинку в памяти через Pillow
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Привязываем мышь к рисованию
        self.canvas.bind("<B1-Motion>", self.paint)

        # 3. Кнопки управления
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.btn_clear = tk.Button(btn_frame, text="Очистить", font=("Arial", 12), command=self.clear)
        self.btn_clear.pack(side=tk.LEFT, padx=10)

        self.btn_predict = tk.Button(btn_frame, text="Распознать", font=("Arial", 12, "bold"), command=self.predict)
        self.btn_predict.pack(side=tk.LEFT, padx=10)

        # 4. Текст с результатом
        self.label_result = tk.Label(root, text="Нарисуйте цифру", font=("Arial", 18, "bold"), fg="blue")
        self.label_result.pack(pady=15)

    def paint(self, event):
        # Толщина кисти (10 пикселей)
        r = 10
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        # Рисуем на экране
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        # Дублируем рисунок в памяти
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Нарисуйте цифру", fg="blue")

    def preprocess(self):
        """Предобработка: приводим рисунок в точный формат MNIST (28x28)"""
        # Находим границы нарисованного (Bounding Box)
        bbox = self.image.getbbox()
        if not bbox:
            return None

        # Вырезаем только саму нарисованную цифру
        cropped = self.image.crop(bbox)

        # Пропорционально сжимаем так, чтобы максимальная сторона была 20 пикселей
        w, h = cropped.size
        if w > h:
            new_w = 20
            new_h = max(1, int(h * (20 / w)))
        else:
            new_h = 20
            new_w = max(1, int(w * (20 / h)))

        resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Вставляем эту цифру ровно в центр черного квадрата 28x28
        final_img = Image.new("L", (28, 28), color=0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_img.paste(resized, (paste_x, paste_y))

        # Переводим в плоский вектор NumPy от 0.0 до 1.0 (длина 784)
        img_array = np.array(final_img, dtype=np.float32) / 255.0
        return img_array.reshape(784)

    def predict(self):
        x_new = self.preprocess()
        if x_new is None:
            self.label_result.config(text="Холст пуст!", fg="red")
            return

        self.label_result.config(text="Думаю...", fg="orange")
        self.root.update()

        # ВЫЗЫВАЕМ ТВОЙ k-NN!
        digit = self.model.predict_one(x_new)

        self.label_result.config(text=f"Это цифра: {digit}", fg="green")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()