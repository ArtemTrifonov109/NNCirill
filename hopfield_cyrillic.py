import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Путь к папке с данными
data_path = r"C:\Neiro\NeiroCirill1.1\CirHop"


# Функция для загрузки изображений из папки
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert('L')
            img = img.resize((28, 28))  # Размер 28x28
            img = np.array(img)
            img = (img > 128).astype(int) * 2 - 1  # Бинаризация: -1 или 1
            images.append(img.flatten())
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
            continue
    return images


# Загрузка всех изображений
all_images = {}
for letter in os.listdir(data_path):
    letter_path = os.path.join(data_path, letter)
    if os.path.isdir(letter_path):
        images = load_images_from_folder(letter_path)
        if len(images) >= 3:  # Убедимся, что есть минимум 3 изображения
            all_images[letter] = images
        else:
            print(f"Папка {letter} содержит менее 3 изображений, пропускаем.")

# Проверка, что изображения загружены
if not all_images:
    raise ValueError("Не удалось загрузить ни одного изображения. Проверьте структуру папки и права доступа.")


# Обучение сети Хопфилда
def train_hopfield(images):
    num_neurons = images[0].shape[0]
    weights = np.zeros((num_neurons, num_neurons))
    for img in images:
        weights += np.outer(img, img)
    np.fill_diagonal(weights, 0)  # Обнуляем диагональ
    return weights


# Функция для добавления шума
def add_noise(image, noise_level):
    noisy_image = image.copy()
    num_noisy = int(noise_level * image.size)
    indices = np.random.choice(image.size, num_noisy, replace=False)
    noisy_image[indices] = -noisy_image[indices]  # Инвертируем пиксели
    return noisy_image


# Функция для восстановления изображения
def recover_image(noisy_image, weights, max_iter=100):
    state = noisy_image.copy()
    for _ in range(max_iter):
        prev_state = state.copy()
        for i in np.random.permutation(len(state)):  # Асинхронное обновление
            state[i] = 1 if np.dot(weights[i], state) >= 0 else -1
        if np.array_equal(state, prev_state):  # Проверка сходимости
            break
    return state


# Уровни шума для экспериментов
noise_levels = [0.2, 0.4, 0.8]

# Визуализация результатов
for letter in all_images:
    # Используем первые два изображения для обучения, третье — для теста
    train_images = all_images[letter][:2]
    test_image = all_images[letter][2]

    # Обучаем сеть только на двух изображениях
    weights = train_hopfield(train_images)

    # Создаем три зашумленные версии одного тестового изображения с разными уровнями шума
    noisy_images = [add_noise(test_image, noise_level=nl) for nl in noise_levels]

    # Восстанавливаем изображения
    recovered_images = [recover_image(noisy, weights) for noisy in noisy_images]

    # Создаем три графика
    for i, noise_level in enumerate(noise_levels):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'Буква {letter} - Уровень шума {noise_level}', fontsize=16)

        # Оригинальное изображение
        axes[0].imshow(test_image.reshape(28, 28), cmap='gray')
        axes[0].set_title('Оригинал')
        axes[0].axis('off')

        # Зашумленное изображение
        axes[1].imshow(noisy_images[i].reshape(28, 28), cmap='gray')
        axes[1].set_title(f'Зашумленное (шум {noise_level})')
        axes[1].axis('off')

        # Восстановленное изображение
        axes[2].imshow(recovered_images[i].reshape(28, 28), cmap='gray')
        axes[2].set_title('Восстановленное')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()