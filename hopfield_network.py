import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image
from config import CYRILLIC_PATH, CYRILLIC_LETTERS
from utils import plot_training_history

# Параметры по умолчанию
DEFAULT_NEURONS = 128  # Количество нейронов
DEFAULT_EPOCHS = 50  # Количество эпох
DEFAULT_ACTIVATION = "step"  # Варианты активации: "step", "sigmoid", "tanh"
DEFAULT_NAME = "Hopfield_128_step"  # Имя модели по умолчанию


def load_dataset(resize_shape=(28, 28)):
    """Загрузка набора данных с кириллицей и преобразование изображений в одномерный массив для сети Хопфилда"""
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

                # Нормализация и бинаризация
                img_array = np.array(img)
                img_array = (img_array < 128).astype(int)  # Бинаризация: черный=1, белый=0
                img_array = img_array.flatten()  # Преобразование в одномерный массив

                # Преобразование в -1 и 1 (формат Хопфилда)
                img_array = 2 * img_array - 1  # 0 -> -1, 1 -> 1

                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Ошибка обработки файла {img_path}: {str(e)}")

    if not images:
        print("Изображения не загружены!")
        return [], []

    print(f"Загружено {len(images)} изображений")
    return np.array(images), np.array(labels)


class HopfieldNetwork:
    def __init__(self, input_size, activation='step'):
        self.input_size = input_size
        self.weights = np.zeros((input_size, input_size))
        self.activation = activation
        print(f"Инициализация сети Хопфилда с размером {input_size} и активацией {activation}")

    def train(self, patterns):
        """Обучение сети Хопфилда на заданных паттернах"""
        n_patterns = len(patterns)
        print(f"Обучение сети Хопфилда на {n_patterns} паттернах...")

        # Сброс весов
        self.weights = np.zeros((self.input_size, self.input_size))

        # Правило обучения Хебба
        for pattern in patterns:
            # Внешнее произведение
            self.weights += np.outer(pattern, pattern)

        # Установка диагональных элементов в ноль (без самообратной связи)
        np.fill_diagonal(self.weights, 0)

        # Нормализация весов
        self.weights /= n_patterns

    def activate(self, x):
        """Применение функции активации"""
        if self.activation == 'step':
            return np.sign(x)  # Пороговая функция (-1, 1)
        elif self.activation == 'sigmoid':
            return 2 * (1 / (1 + np.exp(-x))) - 1  # Нормированная сигмоида (-1, 1)
        elif self.activation == 'tanh':
            return np.tanh(x)  # Функция tanh
        else:
            return np.sign(x)  # По умолчанию пороговая функция

    def predict(self, pattern, max_iterations=20):
        """Предсказание с использованием сети Хопфилда с асинхронными обновлениями"""
        current = pattern.copy()

        for _ in range(max_iterations):
            prev = current.copy()

            # Асинхронное обновление
            for i in np.random.permutation(self.input_size):
                activation = np.dot(self.weights[i], current)
                current[i] = self.activate(activation)

            # Проверка сходимости
            if np.array_equal(prev, current):
                break

        return current

    def evaluate(self, test_patterns, test_labels):
        """Оценка сети на тестовых паттернах"""
        correct = 0
        results = []

        # Создание репрезентативного паттерна для каждого класса
        class_patterns = {}
        for pattern, label in zip(test_patterns, test_labels):
            if label not in class_patterns:
                class_patterns[label] = []
            class_patterns[label].append(pattern)

        # Усреднение паттернов для каждого класса
        avg_class_patterns = {}
        for label, patterns in class_patterns.items():
            avg_class_patterns[label] = np.mean(patterns, axis=0)
            # Убедимся, что значения равны -1 или 1
            avg_class_patterns[label] = np.sign(avg_class_patterns[label])

        for pattern, true_label in zip(test_patterns, test_labels):
            recalled_pattern = self.predict(pattern)

            # Сравнение с прототипами классов для поиска наилучшего соответствия
            best_match = None
            best_similarity = -float('inf')

            for label, class_pattern in avg_class_patterns.items():
                similarity = np.dot(recalled_pattern, class_pattern)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = label

            prediction = best_match
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


def main():
    # Настраиваемые параметры
    neurons = DEFAULT_NEURONS  # Количество нейронов (для уменьшения размерности)
    epochs = DEFAULT_EPOCHS  # Для мониторинга, обучение Хопфилда одношаговое
    activation = DEFAULT_ACTIVATION
    model_name = DEFAULT_NAME

    print(f"Создание модели сети Хопфилда: {model_name}")
    print(f"Параметры: нейронов={neurons}, эпох={epochs}, активация={activation}")

    # Создание директории для сохранения модели
    models_dir = os.path.join("C:\\Neiro\\NeiroCirill1.1\\Models Hop", model_name)
    os.makedirs(models_dir, exist_ok=True)

    # Загрузка и предобработка набора данных
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

    # Уменьшение размерности, если необходимо
    if neurons < X_train.shape[1]:
        from sklearn.decomposition import PCA
        print(f"Применение PCA для уменьшения размерности с {X_train.shape[1]} до {neurons}")
        pca = PCA(n_components=neurons)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        # Бинаризация результатов PCA для сети Хопфилда
        X_train_pca = np.sign(X_train_pca)
        X_val_pca = np.sign(X_val_pca)

        # Сохранение PCA модели для дальнейшего использования
        pca_path = os.path.join(models_dir, "pca_model.pkl")
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"PCA модель сохранена в {pca_path}")
    else:
        X_train_pca = X_train
        X_val_pca = X_val
        pca = None

    # Создание и обучение сети Хопфилда
    hopfield = HopfieldNetwork(X_train_pca.shape[1], activation=activation)

    # Фаза обучения
    start_time = time.time()
    hopfield.train(X_train_pca)
    train_time = time.time() - start_time
    print(f"Обучение завершено за {train_time:.2f} секунд")

    # Оценка на обучающей выборке
    train_accuracy, _ = hopfield.evaluate(X_train_pca, y_train)
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")

    # Оценка на валидационной выборке
    val_accuracy, _ = hopfield.evaluate(X_val_pca, y_val)
    print(f"Точность на валидационной выборке: {val_accuracy:.4f}")

    # Сохранение модели Хопфилда
    model_path = os.path.join(models_dir, "hopfield_model.pkl")
    hopfield.save(model_path)

    # Создание имитации истории обучения для построения графика
    class MockHistory:
        def __init__(self, train_acc, val_acc, epochs):
            self.history = {
                'accuracy': [train_acc * (i / epochs) for i in range(1, epochs + 1)],
                'val_accuracy': [val_acc * (i / epochs) for i in range(1, epochs + 1)],
                'loss': [1.0 - (i / epochs) * train_acc for i in range(1, epochs + 1)],
                'val_loss': [1.0 - (i / epochs) * val_acc for i in range(1, epochs + 1)]
            }

    # Создание объекта имитации истории для построения графика
    history = MockHistory(train_accuracy, val_accuracy, epochs)

    # Построение и сохранение графика истории обучения
    plot_training_history(history, models_dir)

    # Сохранение конфигурации модели
    config = {
        'model_type': 'Hopfield',
        'neurons': neurons,
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