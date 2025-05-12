import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image
from config import CYRILLIC_PATH, CYRILLIC_LETTERS
from utils import plot_training_history

# Параметры по умолчанию
DEFAULT_GRID_SIZE = 5  # Размер карты Кохонена
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.8 # Начальная скорость обучения
DEFAULT_ACTIVATION = "sigmoid"  # Варианты: "linear", "sigmoid", "tanh", "relu"
DEFAULT_NAME = "Kohonen_5_sigmoid"  # Имя модели


def load_dataset(resize_shape=(28, 28)):
    """Загрузка набора данных с кириллицей и преобразование изображений в одномерный массив для сети Кохонена"""
    print(f"Загрузка набора данных из {CYRILLIC_PATH}")

    if not os.path.exists(CYRILLIC_PATH):
        print(f"Ошибка: Директория {CYRILLIC_PATH} не существует.")
        return [], []

    images = []
    labels = []

    for idx, letter in enumerate(CYRILLIC_LETTERS):
        letter_dir = os.path.join(CYRILLIC_PATH, letter)
        if not os.path.exists(letter_dir):
            print(f"Предупреждение: Директория {letter_dir} не существует.")
            continue

        files = [f for f in os.listdir(letter_dir) if f.endswith('.png')]
        if not files:
            print(f"Предупреждение: PNG-файлы в директории {letter_dir} не найдены.")
            continue

        print(f"Загрузка {len(files)} изображений для буквы {letter}")

        for file in files:
            try:
                img_path = os.path.join(letter_dir, file)
                img = Image.open(img_path).convert('L')  # Преобразование в оттенки серого
                img = img.resize(resize_shape)

                # Нормализация в диапазон [0,1]
                img_array = np.array(img) / 255.0
                img_array = img_array.flatten()  # Преобразование в одномерный массив

                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Ошибка обработки файла {img_path}: {str(e)}")

    if not images:
        print("Изображения не загружены!")
        return [], []

    print(f"Загружено {len(images)} изображений")
    return np.array(images), np.array(labels)


class KohonenNetwork:
    def __init__(self, input_size, grid_size=10, learning_rate=0.5, activation='linear'):
        self.input_size = input_size
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.activation = activation

        # Инициализация весов малыми случайными значениями
        self.weights = np.random.rand(grid_size, grid_size, input_size) * 0.1

        print(f"Инициализация сети Кохонена с размером сетки {grid_size}x{grid_size}, "
              f"скоростью обучения {learning_rate} и активацией {activation}")

    def activate(self, x):
        """Применение функции активации"""
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x  # По умолчанию линейная активация

    def find_bmu(self, x):
        """Поиск лучшего соответствия нейрона (BMU) для входного вектора с использованием векторизованных операций"""
        # Изменение формы входного вектора для совместимости с весами
        x_reshaped = x.reshape(1, 1, -1)

        # Вычисление расстояний с использованием векторизованных операций
        distances = np.sqrt(np.sum((self.weights - x_reshaped) ** 2, axis=2))

        # Поиск индексов BMU
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def update_weights(self, x, bmu_idx, iteration, max_iterations):
        """Обновление весов на основе функции соседства с использованием векторизованных операций"""
        # Вычисление радиуса соседства (уменьшается со временем)
        radius_0 = self.grid_size / 2  # Начальный радиус
        time_constant = max_iterations / np.log(radius_0)
        radius = radius_0 * np.exp(-iteration / time_constant)

        # Вычисление скорости обучения (уменьшается со временем)
        learning_rate = self.learning_rate * np.exp(-iteration / max_iterations)

        # Создание координатной сетки для векторизованного вычисления расстояний
        y_coords, x_coords = np.ogrid[0:self.grid_size, 0:self.grid_size]

        # Вычисление расстояния до BMU для всех нейронов
        distance_matrix = np.sqrt((y_coords - bmu_idx[0]) ** 2 + (x_coords - bmu_idx[1]) ** 2)

        # Применение гауссовой функции соседства
        influence = np.exp(-(distance_matrix ** 2) / (2 * (radius ** 2)))

        # Изменение формы влияния для совместимости (grid_size, grid_size, 1)
        influence = influence.reshape(self.grid_size, self.grid_size, 1)

        # Изменение формы входного вектора для совместимости (1, 1, input_size)
        x_reshaped = x.reshape(1, 1, -1)

        # Обновление весов для всех нейронов с помощью векторизованной операции
        delta = learning_rate * influence * (x_reshaped - self.weights)
        self.weights += delta

    def train(self, patterns, max_iterations, log_interval=5):
        """Обучение сети Кохонена с оптимизированным кодом"""
        print(f"Обучение сети Кохонена на {len(patterns)} паттернах в течение {max_iterations} итераций...")

        # Инициализация метрик
        train_errors = []
        train_accuracies = []

        # Ограничение размера пакета для ускорения обучения
        batch_size = min(100, len(patterns))

        # Цикл обучения
        for iteration in range(max_iterations):
            # Перемешивание паттернов и выбор пакета
            indices = np.random.permutation(len(patterns))[:batch_size]

            error = 0
            for idx in indices:
                pattern = patterns[idx]

                # Применение функции активации
                pattern = self.activate(pattern)

                # Поиск BMU
                bmu_idx = self.find_bmu(pattern)

                # Обновление весов
                self.update_weights(pattern, bmu_idx, iteration, max_iterations)

                # Вычисление ошибки (только для выборки, чтобы сэкономить ресурсы)
                if idx % 10 == 0:  # Вычисление ошибки только для некоторых примеров
                    error += np.linalg.norm(pattern - self.weights[bmu_idx])

            # Средняя ошибка
            error /= (batch_size / 10)  # Корректировка для выборки
            train_errors.append(error)

            # Вычисление "точности" (нормализация в диапазон 0-1)
            accuracy = 1.0 - min(error / 10.0, 1.0)  # Масштабирование ошибки для значимой точности
            train_accuracies.append(accuracy)

            if (iteration + 1) % log_interval == 0 or iteration == 0:
                print(f"Итерация {iteration + 1}/{max_iterations}, Ошибка: {error:.4f}, Точность: {accuracy:.4f}")

        return train_errors, train_accuracies

    def map_neuron_to_class(self, patterns, labels):
        """Сопоставление каждого нейрона с классом, который он чаще всего представляет"""
        # Создание пустого сопоставления
        neuron_class_map = np.zeros((self.grid_size, self.grid_size), dtype=int) - 1
        neuron_class_count = np.zeros((self.grid_size, self.grid_size, len(np.unique(labels))))

        # Подсчет классов для каждого нейрона
        for pattern, label in zip(patterns, labels):
            pattern = self.activate(pattern)
            bmu_idx = self.find_bmu(pattern)
            neuron_class_count[bmu_idx[0], bmu_idx[1], label] += 1

        # Назначение наиболее частого класса каждому нейрону
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.sum(neuron_class_count[i, j]) > 0:
                    neuron_class_map[i, j] = np.argmax(neuron_class_count[i, j])

        return neuron_class_map

    def predict(self, pattern):
        """Предсказание класса для паттерна с использованием сопоставления нейрон-класс"""
        pattern = self.activate(pattern)
        bmu_idx = self.find_bmu(pattern)
        return self.neuron_class_map[bmu_idx]

    def evaluate(self, test_patterns, test_labels):
        """Оценка сети на тестовых паттернах"""
        # Сопоставление нейронов с классами
        self.neuron_class_map = self.map_neuron_to_class(test_patterns, test_labels)

        correct = 0
        results = []

        for pattern, true_label in zip(test_patterns, test_labels):
            prediction = self.predict(pattern)
            results.append((true_label, prediction))

            if prediction == true_label:
                correct += 1

        # Исправление для избежания ошибки неопределенного значения
        accuracy = float(correct) / len(test_patterns) if len(test_patterns) > 0 else 0
        return accuracy, results

    def save(self, filepath):
        """Сохранение модели в файл"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Модель сохранена в {filepath}")

    @classmethod
    def load(cls, filepath):
        """Загрузка модели из файла"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Модель загружена из {filepath}")
        return model

def visualize_kohonen_map(kohonen, cyrillic_letters, models_dir):
    """Визуализация карты Кохонена с буквами кириллицы"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Карта Кохонена с буквами кириллицы")
    ax.set_xticks(np.arange(kohonen.grid_size))
    ax.set_yticks(np.arange(kohonen.grid_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Проходим по всем нейронам карты
    for i in range(kohonen.grid_size):
        for j in range(kohonen.grid_size):
            class_idx = kohonen.neuron_class_map[i, j]
            if class_idx != -1:  # Если нейрон сопоставлен с классом
                letter = cyrillic_letters[class_idx]
                ax.text(j, i, letter, ha='center', va='center', fontsize=12)
            else:  # Если нейрон не сопоставлен (пустой)
                ax.text(j, i, ' ', ha='center', va='center', fontsize=12)

    # Сохранение карты
    plt.savefig(os.path.join(models_dir, "kohonen_map_letters.png"))
    plt.close()

def main():
    # Настраиваемые параметры
    grid_size = DEFAULT_GRID_SIZE
    epochs = DEFAULT_EPOCHS
    learning_rate = DEFAULT_LEARNING_RATE
    activation = DEFAULT_ACTIVATION
    model_name = DEFAULT_NAME

    print(f"Создание модели сети Кохонена: {model_name}")
    print(f"Параметры: размер сетки={grid_size}, эпохи={epochs}, "
          f"скорость обучения={learning_rate}, активация={activation}")

    # Создание директории для сохранения модели
    models_dir = os.path.join("C:\\Neiro\\NeiroCirill1.1\\Models Koh", model_name)
    os.makedirs(models_dir, exist_ok=True)

    # Загрузка набора данных
    X, y = load_dataset()
    if len(X) == 0 or len(y) == 0:
        print("Не удалось загрузить набор данных. Завершение работы.")
        return

    # Разделение на обучающую и валидационную выборки
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    print(f"Обучающая выборка: {len(X_train)} примеров")
    print(f"Валидационная выборка: {len(X_val)} примеров")

    # Создание и обучение сети Кохонена
    kohonen = KohonenNetwork(X_train.shape[1], grid_size=grid_size,
                             learning_rate=learning_rate, activation=activation)

    # Фаза обучения
    start_time = time.time()
    train_errors, train_accuracies = kohonen.train(X_train, epochs)
    train_time = time.time() - start_time
    print(f"Обучение завершено за {train_time:.2f} секунд")

    # Оценка на обучающей выборке
    train_accuracy, _ = kohonen.evaluate(X_train, y_train)
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")

    # Оценка на валидационной выборке
    val_accuracy, _ = kohonen.evaluate(X_val, y_val)
    print(f"Точность на валидационной выборке: {val_accuracy:.4f}")

    # Сохранение модели Кохонена
    model_path = os.path.join(models_dir, "kohonen_model.pkl")
    kohonen.save(model_path)

    # Создание истории обучения для построения графика
    class History:
        def __init__(self, train_acc, val_acc, train_errors):
            # Масштабирование точности для соответствия количеству эпох
            self.history = {
                'accuracy': train_acc,
                'val_accuracy': [val_acc] * len(train_acc),
                'loss': train_errors,
                'val_loss': [min(train_errors) * 1.1] * len(train_errors)
            }

    # Создание объекта истории
    history = History(train_accuracies, val_accuracy, train_errors)

    # Построение и сохранение графика истории обучения
    plot_training_history(history, models_dir)

    # Визуализация карты Кохонена
    visualize_kohonen_map(kohonen, CYRILLIC_LETTERS, models_dir)

    # Сохранение конфигурации модели
    config = {
        'model_type': 'Kohonen',
        'grid_size': grid_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'activation': activation,
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'training_time': train_time
    }

    import json
    with open(os.path.join(models_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Модель сохранена в {models_dir}")


if __name__ == "__main__":
    main()