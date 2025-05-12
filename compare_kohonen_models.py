import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import pickle
from PIL import Image
from config import CYRILLIC_LETTERS
# Import the KohonenNetwork class
from kohonen_network import KohonenNetwork


def load_test_data(test_path="C:\\Neiro\\NeiroCirill1.1\\CyrillicTest", resize_shape=(28, 28)):
    """Загрузка тестового набора данных из папки CyrillicTest"""
    print(f"Загрузка тестовых данных из {test_path}")

    if not os.path.exists(test_path):
        print(f"Ошибка: Тестовая директория {test_path} не существует.")
        return [], []

    images = []
    labels = []

    for idx, letter in enumerate(CYRILLIC_LETTERS):
        letter_dir = os.path.join(test_path, letter)
        if not os.path.exists(letter_dir):
            print(f"Предупреждение: Директория {letter_dir} не существует.")
            continue

        files = [f for f in os.listdir(letter_dir) if f.endswith('.png')]
        if not files:
            print(f"Предупреждение: PNG-файлы в директории {letter_dir} не найдены.")
            continue

        print(f"Загрузка {len(files)} тестовых изображений для буквы {letter}")

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
        print("Тестовые изображения не загружены!")
        return [], []

    print(f"Загружено {len(images)} тестовых изображений")
    return np.array(images), np.array(labels)


def compare_models():
    """Сравнение всех моделей Кохонена без переобучения"""
    models_dir = "C:\\Neiro\\NeiroCirill1.1\\Models Koh"
    output_dir = "C:\\Neiro\\NeiroCirill1.1\\AllTest"
    os.makedirs(output_dir, exist_ok=True)

    # Получение списка директорий всех моделей
    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]

    if not model_dirs:
        print("Модели Кохонена не найдены!")
        return

    print(f"Найдено {len(model_dirs)} моделей Кохонена: {model_dirs}")

    # Загрузка тестовых данных
    X_test, y_test = load_test_data()
    if len(X_test) == 0:
        print("Тестовые данные недоступны. Завершение работы.")
        return

    # Сбор результатов
    model_names = []
    accuracies = []
    grid_sizes = []
    activations = []
    learning_rates = []

    # Загрузка моделей и оценка
    for model_name in model_dirs:
        model_path = os.path.join(models_dir, model_name)
        config_path = os.path.join(model_path, 'config.json')
        model_file = os.path.join(model_path, 'kohonen_model.pkl')

        # Пропуск, если файл конфигурации отсутствует
        if not os.path.exists(config_path):
            print(f"Пропуск модели {model_name} - файл конфигурации не найден")
            continue

        # Загрузка конфигурации модели
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"Оценка модели: {model_name}")
        print(f"Конфигурация: {config}")

        # Пропуск моделей, не относящихся к Кохонену
        if config.get('model_type') != 'Kohonen':
            print(f"Пропуск модели {model_name} - не является моделью Кохонена")
            continue

        # Извлечение параметров модели
        grid_size = config.get('grid_size', 10)
        learning_rate = config.get('learning_rate', 0.5)
        activation = config.get('activation', 'linear')

        try:
            # Загрузка сохраненной модели
            if os.path.exists(model_file):
                print(f"Загрузка сохраненной модели из {model_file}")
                with open(model_file, 'rb') as f:
                    kohonen = pickle.load(f)

                # Оценка модели на тестовых данных без переобучения
                accuracy, _ = kohonen.evaluate(X_test, y_test)
                print(f"Точность на тестовых данных: {accuracy:.4f}")
            else:
                print(f"Сохраненная модель не найдена для {model_name}, пропуск...")
                continue

        except Exception as e:
            print(f"Ошибка при оценке модели {model_name}: {str(e)}")
            continue

        # Сохранение результатов
        model_names.append(model_name)
        accuracies.append(accuracy)
        grid_sizes.append(grid_size)
        activations.append(activation)
        learning_rates.append(learning_rate)

    # Создание таблицы результатов
    results = pd.DataFrame({
        'Модель': model_names,
        'Точность': accuracies,
        'РазмерСетки': grid_sizes,
        'Активация': activations,
        'СкоростьОбучения': learning_rates
    })

    # Сортировка по точности
    results = results.sort_values('Точность', ascending=False)

    # Сохранение результатов в CSV
    results_path = os.path.join(output_dir, 'kohonen_models_comparison.csv')
    results.to_csv(results_path, index=False)
    print(f"Результаты сохранены в {results_path}")

    # Построение графика результатов
    if len(results) > 0:  # Проверка наличия результатов
        plt.figure(figsize=(12, 6))
        plt.bar(results['Модель'], results['Точность'], color='lightgreen')
        plt.title('Сравнение точности сетей Кохонена')
        plt.xlabel('Модель')
        plt.ylabel('Точность')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()

        # Добавление значений над столбцами
        for i, v in enumerate(results['Точность']):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

        # Сохранение графика
        plot_path = os.path.join(output_dir, 'kohonen_models_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"График сохранен в {plot_path}")

        # Создание графика зависимости точности от размера сетки и активации
        plt.figure(figsize=(12, 6))

        # Группировка по размеру сетки и активации
        grouped = results.groupby(['РазмерСетки', 'Активация'])['Точность'].mean().reset_index()

        # Создание позиций для группированных столбцов
        grid_sizes_unique = sorted(grouped['РазмерСетки'].unique())
        act_unique = sorted(grouped['Активация'].unique())

        width = 0.2
        x = np.arange(len(grid_sizes_unique))

        # Построение группированных столбцов
        for i, act in enumerate(act_unique):
            act_data = grouped[grouped['Активация'] == act]
            heights = [act_data[act_data['РазмерСетки'] == n]['Точность'].values[0]
                       if n in act_data['РазмерСетки'].values else 0
                       for n in grid_sizes_unique]
            plt.bar(x + i * width - width * (len(act_unique) - 1) / 2,
                    heights, width, label=act)

        plt.xlabel('Размер сетки')
        plt.ylabel('Средняя точность')
        plt.title('Производительность сети Кохонена в зависимости от размера сетки и активации')
        plt.xticks(x, grid_sizes_unique)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()

        # Сохранение графика
        plot_path = os.path.join(output_dir, 'kohonen_performance_by_parameters.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"График анализа параметров сохранен в {plot_path}")

        # Создание графика анализа скорости обучения
        plt.figure(figsize=(12, 6))

        # Группировка по скорости обучения и активации
        grouped_lr = results.groupby(['СкоростьОбучения', 'Активация'])['Точность'].mean().reset_index()

        # Создание позиций для группированных столбцов
        lr_unique = sorted(grouped_lr['СкоростьОбучения'].unique())

        width = 0.2
        x = np.arange(len(lr_unique))

        # Построение группированных столбцов
        for i, act in enumerate(act_unique):
            act_data = grouped_lr[grouped_lr['Активация'] == act]
            heights = [act_data[act_data['СкоростьОбучения'] == n]['Точность'].values[0]
                       if n in act_data['СкоростьОбучения'].values else 0
                       for n in lr_unique]
            plt.bar(x + i * width - width * (len(act_unique) - 1) / 2,
                    heights, width, label=act)

        plt.xlabel('Скорость обучения')
        plt.ylabel('Средняя точность')
        plt.title('Производительность сети Кохонена в зависимости от скорости обучения и активации')
        plt.xticks(x, lr_unique)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()

        # Сохранение графика
        plot_path = os.path.join(output_dir, 'kohonen_performance_by_learning_rate.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"График анализа скорости обучения сохранен в {plot_path}")

        # Вывод итогов
        print("\nИтоги:")
        if len(results) > 0:
            print(f"Лучшая модель: {results.iloc[0]['Модель']} с точностью {results.iloc[0]['Точность']:.4f}")
            if len(results) > 1:
                print(f"Худшая модель: {results.iloc[-1]['Модель']} с точностью {results.iloc[-1]['Точность']:.4f}")
        else:
            print("Нет доступных результатов для анализа")
    else:
        print("Нет результатов для построения графиков")

    return results


if __name__ == "__main__":
    compare_models()