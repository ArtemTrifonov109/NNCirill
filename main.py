import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from config import (
    PROJECT_ROOT, MODELS_PATH, STATISTICS_PATH, CYRILLIC_PATH,
    ACTIVATION_FUNCTIONS, OPTIMIZERS, HIDDEN_UNITS_OPTIONS,
    HIDDEN_LAYERS_OPTIONS, DEFAULT_PARAMETERS, IMG_WIDTH, IMG_HEIGHT, CYRILLIC_LETTERS
)
from utils import (
    load_dataset, convert_drawn_image, save_model_config,
    load_model_config, get_letter_prediction, plot_training_history
)
from models import train_models, load_models
from statistics import Statistics

# Настраиваем TensorFlow для использования GPU, если доступен
print("Проверка доступности GPU...")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Найден GPU: {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Включен режим динамического выделения памяти для GPU")
else:
    print("GPU не обнаружен, будет использоваться CPU")


class DrawingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.drawing = False
        self.points = []
        self.last_x = 0
        self.last_y = 0

        self.image = Image.new("L", (278, 278), color=255)
        self.draw_image = ImageDraw.Draw(self.image)

        # Привязываем события при инициализации, но канвас будет активирован позже
        self.bind("<Button-1>", self.start_drawing)
        self.bind("<B1-Motion>", self.draw_line)
        self.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.points = []
        self.points.append((event.x, event.y))
        # Вызываем предсказание TDNN если оно доступно
        if hasattr(self.parent, 'predict_tdnn_realtime'):
            self.parent.predict_tdnn_realtime()

    def draw_line(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.create_line((self.last_x, self.last_y, x, y), width=15, fill="black",
                             capstyle=tk.ROUND, smooth=True, splinesteps=36)
            self.draw_image.line([self.last_x, self.last_y, x, y], fill=0, width=15)
            self.last_x = x
            self.last_y = y
            self.points.append((x, y))
            # Вызываем предсказание TDNN при каждом движении
            if hasattr(self.parent, 'predict_tdnn_realtime'):
                self.parent.predict_tdnn_realtime()

    def stop_drawing(self, event):
        self.drawing = False
        self.points.append((event.x, event.y))

    def clear(self):
        self.delete("all")
        self.image = Image.new("L", (278, 278), color=255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.points = []
        if hasattr(self.parent, 'clear_predictions'):
            self.parent.clear_predictions()

    def get_image(self):
        return self.image


class NeiroCirillApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeiroCirill")
        self.geometry("1000x600")
        self.configure(bg="#f5f5f7")

        self.ffnn_model = None
        self.tdnn_model = None
        self.current_model_dir = None
        self.statistics = None

        self.create_widgets()
        self.check_directories()

    def enable_drawing(self):
        """Активирует канвас для рисования"""
        self.canvas.config(state="normal")
        self.status_var.set("Модель загружена. Можно рисовать.")

    def disable_drawing(self):
        """Деактивирует канвас для рисования"""
        self.canvas.config(state="disabled")

    def check_directories(self):
        for path in [PROJECT_ROOT, MODELS_PATH, STATISTICS_PATH]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        # Проверяем наличие директории для букв
        if not os.path.exists(CYRILLIC_PATH):
            messagebox.showwarning("Предупреждение",
                                   "Директория для данных не найдена. Сначала создайте папки с изображениями букв.")

    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        drawing_frame = ttk.LabelFrame(main_frame, text="Рисование буквы")
        drawing_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.canvas = DrawingCanvas(self, width=278, height=278, bg="white", highlightthickness=1)
        self.canvas.pack(in_=drawing_frame, padx=10, pady=10)
        self.canvas.config(state="disabled")  # Канвас изначально неактивен

        clear_btn = ttk.Button(drawing_frame, text="Очистить", command=self.canvas.clear)
        clear_btn.pack(pady=5)

        config_frame = ttk.LabelFrame(main_frame, text="Настройки модели")
        config_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        inner_frame = ttk.Frame(config_frame)
        inner_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(inner_frame, text="Функция активации:").grid(row=0, column=0, sticky="w", pady=5)
        self.activation_var = tk.StringVar(value=DEFAULT_PARAMETERS["activation"])
        activation_combo = ttk.Combobox(inner_frame, textvariable=self.activation_var,
                                        values=ACTIVATION_FUNCTIONS, state="readonly")
        activation_combo.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(inner_frame, text="Оптимизатор:").grid(row=1, column=0, sticky="w", pady=5)
        self.optimizer_var = tk.StringVar(value=DEFAULT_PARAMETERS["optimizer"])
        optimizer_combo = ttk.Combobox(inner_frame, textvariable=self.optimizer_var,
                                       values=OPTIMIZERS, state="readonly")
        optimizer_combo.grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(inner_frame, text="Число элементов (Ns):").grid(row=2, column=0, sticky="w", pady=5)
        self.hidden_units_var = tk.IntVar(value=DEFAULT_PARAMETERS["hidden_units"])
        hidden_units_combo = ttk.Combobox(inner_frame, textvariable=self.hidden_units_var,
                                          values=HIDDEN_UNITS_OPTIONS, state="readonly")
        hidden_units_combo.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(inner_frame, text="Число скрытых слоев:").grid(row=3, column=0, sticky="w", pady=5)
        self.hidden_layers_var = tk.IntVar(value=DEFAULT_PARAMETERS["hidden_layers"])
        hidden_layers_combo = ttk.Combobox(inner_frame, textvariable=self.hidden_layers_var,
                                           values=HIDDEN_LAYERS_OPTIONS, state="readonly")
        hidden_layers_combo.grid(row=3, column=1, sticky="ew", pady=5)

        ttk.Label(inner_frame, text="Число эпох:").grid(row=4, column=0, sticky="w", pady=5)
        self.epochs_var = tk.IntVar(value=DEFAULT_PARAMETERS["epochs"])
        epochs_entry = ttk.Entry(inner_frame, textvariable=self.epochs_var)
        epochs_entry.grid(row=4, column=1, sticky="ew", pady=5)

        ttk.Label(inner_frame, text="Размер пакета:").grid(row=5, column=0, sticky="w", pady=5)
        self.batch_size_var = tk.IntVar(value=DEFAULT_PARAMETERS["batch_size"])
        batch_size_entry = ttk.Entry(inner_frame, textvariable=self.batch_size_var)
        batch_size_entry.grid(row=5, column=1, sticky="ew", pady=5)

        # Removed the test_split parameter UI element

        button_frame = ttk.Frame(inner_frame)
        button_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=10)

        train_btn = ttk.Button(button_frame, text="Обучить", command=self.train_model)
        train_btn.grid(row=0, column=0, padx=5)

        load_btn = ttk.Button(button_frame, text="Загрузить модель", command=self.load_model)
        load_btn.grid(row=0, column=1, padx=5)

        ffnn_frame = ttk.LabelFrame(main_frame, text="Ответ нейросети FFNN")
        ffnn_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.ffnn_result_var = tk.StringVar(value="")
        ttk.Label(ffnn_frame, textvariable=self.ffnn_result_var, font=("Arial", 14, "bold")).pack(pady=10)

        self.ffnn_confidence_frame = ttk.Frame(ffnn_frame)
        self.ffnn_confidence_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.ffnn_confidence_frame, text="Уверенность:").pack(side=tk.LEFT)
        self.ffnn_confidence_var = tk.DoubleVar(value=0)
        self.ffnn_confidence_bar = ttk.Progressbar(self.ffnn_confidence_frame,
                                                   variable=self.ffnn_confidence_var,
                                                   length=200, mode="determinate")
        self.ffnn_confidence_bar.pack(side=tk.LEFT, padx=5)
        self.ffnn_confidence_label = ttk.Label(self.ffnn_confidence_frame, text="0%")
        self.ffnn_confidence_label.pack(side=tk.LEFT)

        recognize_btn = ttk.Button(ffnn_frame, text="Определить", command=self.predict_ffnn)
        recognize_btn.pack(pady=10)

        tdnn_frame = ttk.LabelFrame(main_frame, text="Ответ нейросети TDNN")
        tdnn_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        self.tdnn_results_frame = ttk.Frame(tdnn_frame)
        self.tdnn_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tdnn_result_widgets = []

        self.status_var = tk.StringVar(value="Готов к работе. Пожалуйста, загрузите или обучите модель.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def train_model(self):
        config = {
            "activation": self.activation_var.get(),
            "optimizer": self.optimizer_var.get(),
            "hidden_units": self.hidden_units_var.get(),
            "hidden_layers": self.hidden_layers_var.get(),
            "epochs": self.epochs_var.get(),
            "batch_size": self.batch_size_var.get()
            # Removed test_split parameter
        }

        model_name = self.prompt_model_name()
        if not model_name:
            return

        model_dir = os.path.join(MODELS_PATH, model_name)
        if os.path.exists(model_dir):
            overwrite = messagebox.askyesno("Предупреждение",
                                            f"Модель '{model_name}' уже существует. Перезаписать?")
            if not overwrite:
                return

        os.makedirs(model_dir, exist_ok=True)
        self.statistics = Statistics(model_name)

        try:
            self.status_var.set("Загрузка данных...")
            # Use updated load_dataset function that returns generators
            X_train, y_train, X_test, y_test, train_generator, validation_generator = load_dataset()

            if len(X_train) == 0:
                messagebox.showerror("Ошибка", "Не удалось загрузить данные. Проверьте папку Cyrillic.")
                self.status_var.set("Ошибка загрузки данных.")
                return

            self.status_var.set("Обучение моделей...")
            start_time = time.time()

            # Pass generators to train_models
            self.ffnn_model, self.tdnn_model, ffnn_history, tdnn_history = train_models(
                X_train, y_train, X_test, y_test, config, model_dir,
                train_generator=train_generator, validation_generator=validation_generator
            )

            training_time = time.time() - start_time

            save_model_config(model_dir, config)

            plot_training_history(ffnn_history, os.path.join(model_dir, 'ffnn'))
            plot_training_history(tdnn_history, os.path.join(model_dir, 'tdnn'))

            self.statistics.record_training_stats(ffnn_history, tdnn_history, config, training_time)

            self.current_model_dir = model_dir
            self.enable_drawing()  # Активируем холст после обучения

            self.status_var.set(f"Модели успешно обучены и сохранены в '{model_name}'")
            messagebox.showinfo("Обучение завершено",
                                f"Модели успешно обучены и сохранены в\n{model_dir}")

        except Exception as e:
            messagebox.showerror("Ошибка обучения", str(e))
            self.status_var.set("Ошибка обучения моделей.")
            print(f"Ошибка обучения: {str(e)}")
            import traceback
            traceback.print_exc()

    def load_model(self):
        # ... keep existing code (загрузка модели)
        model_dir = filedialog.askdirectory(initialdir=MODELS_PATH, title="Выберите папку с моделями")
        if not model_dir:
            return

        try:
            self.status_var.set("Загрузка моделей...")
            print(f"Попытка загрузки моделей из {model_dir}")

            self.ffnn_model, self.tdnn_model = load_models(model_dir)

            if self.ffnn_model is None or self.tdnn_model is None:
                print("Не удалось загрузить обе модели")
                messagebox.showerror("Ошибка",
                                     "Не удалось загрузить модели. Проверьте формат и наличие файлов моделей.")
                self.status_var.set("Ошибка загрузки моделей.")
                return

            print("Обе модели успешно загружены")
            print(f"FFNN модель: {self.ffnn_model}")
            print(f"TDNN модель: {self.tdnn_model}")

            # Загружаем конфигурацию модели если она есть
            config = load_model_config(model_dir)
            if config:
                print(f"Конфигурация модели загружена: {config}")
                self.activation_var.set(config.get("activation", DEFAULT_PARAMETERS["activation"]))
                self.optimizer_var.set(config.get("optimizer", DEFAULT_PARAMETERS["optimizer"]))
                self.hidden_units_var.set(config.get("hidden_units", DEFAULT_PARAMETERS["hidden_units"]))
                self.hidden_layers_var.set(config.get("hidden_layers", DEFAULT_PARAMETERS["hidden_layers"]))
                self.epochs_var.set(config.get("epochs", DEFAULT_PARAMETERS["epochs"]))
                self.batch_size_var.set(config.get("batch_size", DEFAULT_PARAMETERS["batch_size"]))
                self.test_split_var.set(config.get("test_split", DEFAULT_PARAMETERS["test_split"]))
            else:
                print("Конфигурация модели не найдена")

            self.current_model_dir = model_dir
            model_name = os.path.basename(model_dir)
            self.statistics = Statistics(model_name)

            self.enable_drawing()  # Активируем холст после загрузки

            self.status_var.set(f"Модели загружены из '{model_name}'")
            messagebox.showinfo("Загрузка завершена", f"Модели успешно загружены из\n{model_dir}")

        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            self.status_var.set("Ошибка загрузки моделей.")
            print(f"Ошибка загрузки моделей: {str(e)}")
            import traceback
            traceback.print_exc()

    def predict_ffnn(self):
        if self.ffnn_model is None:
            messagebox.showerror("Ошибка", "Модель FFNN не загружена.")
            return

        img = self.canvas.get_image()
        input_img = convert_drawn_image(img)

        start_time = time.time()

        try:
            print("Запуск предсказания FFNN...")
            predictions = self.ffnn_model.predict(input_img, verbose=0)[0]
            prediction_idx = np.argmax(predictions)
            predicted_letter = CYRILLIC_LETTERS[prediction_idx]
            confidence = predictions[prediction_idx] * 100

            response_time = time.time() - start_time

            print(f"Результат FFNN: буква {predicted_letter}, уверенность {confidence:.1f}%")
            self.ffnn_result_var.set(f"Буква: {predicted_letter}")
            self.ffnn_confidence_var.set(confidence)
            self.ffnn_confidence_label.config(text=f"{confidence:.1f}%")

            if self.statistics:
                self.statistics.record_prediction(
                    model_type="FFNN",
                    true_letter=None,
                    predicted_letter=predicted_letter,
                    confidence=confidence,
                    response_time=response_time
                )

        except Exception as e:
            messagebox.showerror("Ошибка распознавания", str(e))
            print(f"Ошибка распознавания FFNN: {str(e)}")
            import traceback
            traceback.print_exc()

    def predict_tdnn_realtime(self):
        if self.tdnn_model is None:
            return

        img = self.canvas.get_image()
        if len(self.canvas.points) < 3:  # Требуем минимум 3 точки для начала распознавания
            return

        input_img = convert_drawn_image(img)

        try:
            start_time = time.time()
            predictions = self.tdnn_model.predict(input_img, verbose=0)[0]
            top_indices = np.argsort(predictions)[-5:][::-1]  # Топ-5 индексов
            results = []

            # Формируем результаты с буквой и уверенностью
            for idx in top_indices:
                if idx < len(CYRILLIC_LETTERS):
                    letter = CYRILLIC_LETTERS[idx]
                    confidence = predictions[idx] * 100
                    results.append((letter, confidence))

            # Обновляем UI с результатами
            self.update_tdnn_results(results)

            # Записываем статистику для топового предсказания
            if results and self.statistics:
                response_time = time.time() - start_time
                self.statistics.record_prediction(
                    model_type="TDNN",
                    true_letter=None,
                    predicted_letter=results[0][0],  # Лучшее предсказание
                    confidence=results[0][1],  # Уверенность для лучшего предсказания
                    response_time=response_time
                )

        except Exception as e:
            print(f"Error in TDNN prediction: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_tdnn_results(self, results):
        # Очищаем предыдущие виджеты
        for widget in self.tdnn_result_widgets:
            widget.destroy()
        self.tdnn_result_widgets = []

        # Создаем новые виджеты для каждого результата
        for i, (letter, confidence) in enumerate(results):
            frame = ttk.Frame(self.tdnn_results_frame)
            frame.pack(fill=tk.X, pady=2)

            # Буква
            label = ttk.Label(frame, text=f"{letter}:", width=3, font=("Arial", 12, "bold"))
            label.pack(side=tk.LEFT, padx=5)

            # Индикатор уверенности
            progress = ttk.Progressbar(frame, length=150, mode="determinate", value=confidence)
            progress.pack(side=tk.LEFT, padx=5)

            # Процент уверенности
            conf_label = ttk.Label(frame, text=f"{confidence:.1f}%")
            conf_label.pack(side=tk.LEFT, padx=5)

            # Сохраняем ссылки на виджеты для последующего удаления
            self.tdnn_result_widgets.extend([frame, label, progress, conf_label])

    def clear_predictions(self):
        self.ffnn_result_var.set("")
        self.ffnn_confidence_var.set(0)
        self.ffnn_confidence_label.config(text="0%")

        for widget in self.tdnn_result_widgets:
            widget.destroy()
        self.tdnn_result_widgets = []

    def prompt_model_name(self):
        model_name = simpledialog.askstring(
            "Имя модели",
            "Введите имя для модели:",
            initialvalue=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        return model_name


if __name__ == "__main__":
    app = NeiroCirillApp()
    app.mainloop()

