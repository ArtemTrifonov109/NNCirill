import os
from PIL import Image, ImageOps, ImageFilter

# Исходный каталог датасета
input_dir = r"C:\Neiro\NeiroCirill1.1\Cyrillic"

# Рекурсивный обход всех подкаталогов
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".png"):
            file_path = os.path.join(root, file)
            try:
                # Открываем изображение
                img = Image.open(file_path)

                # Если изображение имеет альфа-канал, композитим его на белом фоне
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background

                # Преобразуем в градации серого
                img = img.convert('L')

                # Инвертируем изображение для вычисления ограничивающего прямоугольника символа
                inverted = ImageOps.invert(img)
                bbox = inverted.getbbox()
                if bbox is None:
                    print(f"Пустое изображение: {file_path}")
                    continue

                # Обрезаем изображение по ограничивающему прямоугольнику
                cropped = img.crop(bbox)

                # Создаем квадратное изображение с белым фоном и отступами для центрирования символа
                margin = 8  # значение можно корректировать экспериментально
                size_val = max(cropped.width, cropped.height) + margin
                square = Image.new('L', (size_val, size_val), 255)
                x_offset = (size_val - cropped.width) // 2
                y_offset = (size_val - cropped.height) // 2
                square.paste(cropped, (x_offset, y_offset))

                # Масштабируем до 28x28 с использованием Resampling.LANCZOS
                resized = square.resize((28, 28), resample=Image.Resampling.LANCZOS)

                # Применяем фильтр повышения резкости (UnsharpMask) для устранения "мыльности"
                sharpened = resized.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

                # Перезаписываем оригинальный файл
                sharpened.save(file_path)
                print(f"Обработан файл: {file_path}")
            except Exception as e:
                print(f"Ошибка при обработке {file_path}: {e}")

print("Обработка датасета завершена!")
