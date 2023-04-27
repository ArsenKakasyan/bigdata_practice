from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

data = []
labels = []
ParentDir = os.path.abspath('Cyrillic')
ChildDirs = os.listdir(ParentDir)

# Загрузите набор данных с помощью библиотеки обработки изображений, такой как OpenCV или PIL.
for folder in ChildDirs:
    label = folder
    child_folder_path = os.path.join(ParentDir, folder)

    for img_name in os.listdir(child_folder_path):
        img_path = os.path.join(child_folder_path, img_name)

        img = Image.open(img_path)
        
        if img is not None:
            img = img.convert('L')
            data.append(img)
            labels.append(label)
        else:
            print(f"Failed to read image: {img_path}")
# Предварительно обработайте изображения, изменив их размер до меньшего размера, нормализовав значения пикселей и преобразовав метки в однократное кодирование:
# Изменение размера изображений до 28x28
data = [np.array(img.resize((28, 28), resample=Image.BICUBIC)) for img in data]


# Нормализация значений пикселей до [0, 1]
data = np.array(data) / 255.0

# Преобразование меток в однократное кодирование
unique_labels = list(set(labels))
label_dict = {label: i for i, label in enumerate(unique_labels)}
labels = [label_dict[label] for label in labels]
labels = np.eye(len(unique_labels))[labels]

# Разделите набор данных на наборы для обучения и проверки:
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# CNN являются наиболее часто используемой архитектурой для задач классификации изображений:
'''
Это создает простую CNN с двумя сверточными слоями, двумя максимальными объединяющими слоями, полностью связанным слоем со 128 единицами, отсевающим слоем для регуляризации,
и последний выходной слой с активацией softmax.
'''
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(unique_labels), activation='softmax')
])

# Скомпилируйте модель с соответствующей функцией потерь, оптимизатором и метриками:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучите модель на тренировочном наборе, используя метод подгонки:
history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

# Оцените модель в наборе проверки, используя метод оценки:
loss, accuracy = model.evaluate(val_data, val_labels)
print(f"Validation loss: {loss:.4f}, accuracy: {accuracy:.4f}")


# Сохраните веса в файл
model.save_weights('cyrillic_weights.h5')

# Загрузите веса из файла
#model.load_weights('cyrillic_weights.h5')