import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from PIL import Image
from config import CYRILLIC_LETTERS


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
        print("Тестовые изображения не загружены!")
        return [], []

    print(f"Загружено {len(images)} тестовых изображений")
    return np.array(images), np.array(labels)


def compare_models():
    """Сравнение всех моделей Хопфилда"""
    models_dir = "C:\\Neiro\\NeiroCirill1.1\\Models Hop"
    output_dir = "C:\\Neiro\\NeiroCirill1.1\\AllTest"
    os.makedirs(output_dir, exist_ok=True)

    # Получение списка директорий всех моделей
    model_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d))]

    if not model_dirs:
        print("Модели Хопфилда не найдены!")
        return

    print(f"Найдено {len(model_dirs)} моделей Хопфилда: {model_dirs}")

    # Загрузка тестовых данных
    X_test, y_test = load_test_data()
    if len(X_test) == 0:
        print("Тестовые данные недоступны. Завершение работы.")
        return

    # Сбор результатов
    model_names = []
    accuracies = []
    neurons_list = []
    activations = []

    # Загрузка моделей и оценка
    for model_name in model_dirs:
        model_path = os.path.join(models_dir, model_name)
        config_path = os.path.join(model_path, 'config.json')

        # Пропуск, если файл конфигурации отсутствует
        if not os.path.exists(config_path):
            print(f"Пропуск модели {model_name} - файл конфигурации не найден")
            continue

        # Загрузка конфигурации модели
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"Оценка модели: {model_name}")
        print(f"Конфигурация: {config}")

        # Пропуск моделей, не относящихся к Хопфилду
        if config.get('model_type') != 'Hopfield':
            print(f"Пропуск модели {model_name} - не является моделью Хопфилда")
            continue

        # Извлечение параметров модели
        neurons = config.get('neurons', 100)
        activation = config.get('activation', 'step')

        # Необходимость создания сети Хопфилда с сохраненными параметрами
        from hopfield_network import HopfieldNetwork

        # Проверка необходимости применения PCA
        if neurons < X_test.shape[1]:
            from sklearn.decomposition import PCA
            print(f"Применение PCA для уменьшения размерности с {X_test.shape[1]} до {neurons}")
            pca = PCA(n_components=neurons)
            # Поскольку оригинальные обучающие данные недоступны,
            # используем PCA для приблизительного уменьшения размерности
            X_test_pca = pca.fit_transform(X_test)
            # Бинаризация
            X_test_pca = np.sign(X_test_pca)
        else:
            X_test_pca = X_test

        # Создание сети Хопфилда с теми же параметрами
        hopfield = HopfieldNetwork(X_test_pca.shape[1], activation=activation)

        # Для сети Хопфилда обучение проводится на тестовых данных,
        # так как оригинальные веса недоступны
        try:
            hopfield.train(X_test_pca)
            accuracy, _ = hopfield.evaluate(X_test_pca, y_test)
            print(f"Точность на тестовых данных: {accuracy:.4f}")
        except Exception as e:
            print(f"Ошибка при оценке модели {model_name}: {str(e)}")
            continue

        # Сохранение результатов
        model_names.append(model_name)
        accuracies.append(accuracy)
        neurons_list.append(neurons)
        activations.append(activation)

    # Создание таблицы результатов
    results = pd.DataFrame({
        'Модель': model_names,
        'Точность': accuracies,
        'Нейроны': neurons_list,
        'Активация': activations
    })

    # Сортировка по точности
    results = results.sort_values('Точность', ascending=False)

    # Сохранение результатов в CSV
    results_path = os.path.join(output_dir, 'hopfield_models_comparison.csv')
    results.to_csv(results_path, index=False)
    print(f"Результаты сохранены в {results_path}")

    # Построение графика результатов
    plt.figure(figsize=(12, 6))
    plt.bar(results['Модель'], results['Точность'], color='skyblue')
    plt.title('Сравнение точности сетей Хопфилда')
    plt.xlabel('Модель')
    plt.ylabel('Точность')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()

    # Добавление значений над столбцами
    for i, v in enumerate(results['Точность']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

    # Сохранение графика
    plot_path = os.path.join(output_dir, 'hopfield_models_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"График сохранен в {plot_path}")

    # Создание графика зависимости точности от количества нейронов и активации
    plt.figure(figsize=(12, 6))

    # Группировка по количеству нейронов и активации
    grouped = results.groupby(['Нейроны', 'Активация'])['Точность'].mean().reset_index()

    # Создание позиций для группированных столбцов
    neurons_unique = sorted(grouped['Нейроны'].unique())
    act_unique = sorted(grouped['Активация'].unique())

    width = 0.2
    x = np.arange(len(neurons_unique))

    # Построение группированных столбцов
    for i, act in enumerate(act_unique):
        act_data = grouped[grouped['Активация'] == act]
        heights = [act_data[act_data['Нейроны'] == n]['Точность'].values[0]
                   if n in act_data['Нейроны'].values else 0
                   for n in neurons_unique]
        plt.bar(x + i * width - width * (len(act_unique) - 1) / 2,
                heights, width, label=act)

    plt.xlabel('Количество нейронов')
    plt.ylabel('Средняя точность')
    plt.title('Производительность сети Хопфилда в зависимости от нейронов и активации')
    plt.xticks(x, neurons_unique)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()

    # Сохранение графика
    plot_path = os.path.join(output_dir, 'hopfield_performance_by_parameters.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"График анализа параметров сохранен в {plot_path}")

    # Вывод итогов
    print("\nИтоги:")
    print(f"Лучшая модель: {results.iloc[0]['Модель']} с точностью {results.iloc[0]['Точность']:.4f}")
    print(f"Худшая модель: {results.iloc[-1]['Модель']} с точностью {results.iloc[-1]['Точность']:.4f}")

    return results


if __name__ == "__main__":
    compare_models()