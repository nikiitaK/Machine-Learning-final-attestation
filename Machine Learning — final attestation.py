#!/usr/bin/env python
# coding: utf-8

# # Промежуточная аттестация по сетям искусcтвенных нейронов.
# CIFAR-10 содержит 100 классов, по 600 изображений в каждом (500 обучающих + 100 тестовых). Все 100 классов сгруппированы в 20 суперклассов. Изображения имеют размерность 32x32x3. Каждое изображение снабжено "узкой" меткой (класс, к которому оно принадлежит) и "широкой" меткой (суперкласс, к которому оно принадлежит). 
# 
# ## Задание
# - Объяснить, какие элементы вашей сети зависят от количества цветов, какие — от количества классов.
# - Обучить модель. Объяснить место в модели каждого слоя, обосновать выбор гиперпараметров. 
# - Сравнить качество предсказания при обучении на 20 широких классах с предсказаниями при обучении на 100 узких классах,обобщив предсказания по узким меткам до метки их широкого класса.
# - Исследовать с помощью графиков метрики предсказания для каких узких классов более всего отличаются от метрик их более широких классов. Выдвинуть предположение о причине возможного отличия.
# - Сохранить .ipynb файл с объяснениями, оформленными в отдельных текстовых блоках.
# - Преобразовать .ipynb в  программу на Питоне (.py) так, чтобы после обучения модели онаавтоматически экспортировалась бы в файл

#<font size='5' color='#FF0000'> В задании говорилось, что необходимо загрузить результат работы, на GitLab, но сделать этого я не смог(так как для регистрации там требуется номер зарубежного мобильного или карты), поэтому загрузил на GitHub </font>

# ## 1. Подключение библиотек

# In[1]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import random
import tensorflow as tf


# ## 2. Загрузка данных

# In[2]:


(X_train, Y_train_small), (X_test, useless) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
(useless, Y_train_big), (useless, Y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')


# **Так как изображения в CIFAR-100 идентичны для узких и широких классов, я буду использовать лишь по одному набору X_train, X_test, Y_test. Порядок работы будет следующим:**
# <br>Шаг 1: обучу модель на X_train и Y_train_big 
# <br>Шаг 2: используя X_test сделаю предсказание predictions_by_big и сравню его с реальным Y_test
# <br>Шаг 3: обучу модель на X_train и Y_train_small 
# <br>Шаг 4: используя X_test сделаю предсказание predictions_by_small и сравню его с реальным Y_test
# <br>Шаг 5: Сравню predictions_by_big и predictions_by_small

# In[3]:


# Пропишу списки имён классов
names_big = ["aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
             "household electrical devices", "household furniture", "insects", "large carnivores", 
             "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores", 
             "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals", 
             "trees", "vehicles 1", "vehicles 2"]

names_small =['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
              'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
              'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
              'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
              'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
              'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
              'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
              'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
              'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
              'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
              'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
              'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
              'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
              'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
              'worm']

print("count big names:", len(names_big))
print("count small names:", len(names_small))


# ## Визаулизация датасетов с узким и широким классом

# In[4]:


plt.figure(figsize=(20,10))
random_inds = np.random.choice(50000,36)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(X_train[image_ind]))
    plt.xlabel(names_small[Y_train_small[image_ind][0]])


# In[5]:


plt.figure(figsize=(20,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(X_train[image_ind]))
    plt.xlabel(names_big[Y_train_big[image_ind][0]])


# ## 3. Предобработка

# **Масштабирую цвета X_train, X_test в диапазон от 0 до 1 (так как каждый из трёх каналов состоит из 256 оттенков)**

# In[6]:


X_train = (X_train/255.).astype(np.float32)
X_test = (X_test/255.).astype(np.float32)


# **Изменю форму массивов X_train и X_test (batch_size, height, width, сolors)**

# In[7]:


X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# In[8]:


print("X_test shape: ", X_test.shape)
print("X_train shape: ", X_train.shape)


# **Сделаю one_hot преобразование для y_train и y_test**

# In[9]:


Y_train_small = tf.keras.utils.to_categorical(
    Y_train_small, num_classes=len(names_small), dtype='float32'
)
Y_train_big = tf.keras.utils.to_categorical(
    Y_train_big, num_classes=len(names_big), dtype='float32'
)
Y_test = tf.keras.utils.to_categorical(
    Y_test, num_classes=len(names_big), dtype='float32'
)


# **Объясню, какие элементы сети зависят от количества цветов, какие — от количества классов**
# - От количества цветов зависит размерность массива с элементами данных, которая расчитывается по формуле: "количество_элементов x ширина x высота x цветность".
# - От количества классов зависит количество выходных нейронов, необходимых для выделения этих классов.
# 

# ## 4. Создание модели для предсказания широких классов

# Для создания сети из последовательных слоёв используется модель Sequential(). Которая содержит следующие слои:
# 
# **Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3))** - cвёрточный слой с 32 фильтрами и ядром 3x3. 
# <br>Функция активации "relu"- max(0,x) преобразует входное значение в 0 (если значение отрицательно), либо в положительное значение (в остальных случаях). 
# <br>Параметр input_shape служит для описания формы входных данных, в данном случае (32,32,3) - ВЫСОТА x ШИРИНА x ЦВЕТНОСТЬ. <br>Свёрточный слой Conv2D применяется к двухмерным данным для выделения признаков обьектов, он применяет на данные 32 фильтра, а также использует обединение (3,3).
# 
# **MaxPooling2D(pool_size=(2, 2))** - используются для уменьшения размерности входных данных путем выбора максимального значения из окна (2, 2), тоесть слой оставляет лишь четверть значений от исходного. Это позволяет уменьшить количество параметров модели и улучшить ее обобщающую способность.
# 
# Дальше конструкция свёртка+пулинг повторяется ещё два раза, это позволяет отобрать необходимые характеристики, сохранив при этом достаточно низкую размерность.
# 
# **Flatten()** - выпрямляющий слой, который преобразует формат изображений из 2d-массива (128 штук по 2x2 пикселей) в 1d-массив из 128 * 2 * 2 = 512 пикселей
# 
# **Dense(256, activation='relu')** и **Dense(128, activation='relu')** - два полносвязных слоя, необходимых для уменьшения размерности данных с 512 до 128.
# 
# **Dense(20, activation='softmax')** - выходной полносвязный слой на 20 нейронов, проводит категоризацию на 20 выходных категорий. В нём используется функция активации 'softmax', она масштабирует значения нейронов к диапазону от 0 до 1 (чтобы при этом их сумма была равна 1). Таким образом 'softmax' возвращает матрицу вероятностей принадлежности объекта к каждому из классов. Эта функция часто, используется на последнем слою сети.

# In[10]:


def build_cnn_model():
    cnn_model = tf.keras.Sequential([
                                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
                                    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
                                    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(20, activation='softmax') 
    ])

    return cnn_model

model = build_cnn_model()

# Единичный вызов модели для её инициализации
model.predict(X_train[[0]])

# Вывод характеристик модели
print(model.summary())


# Другими гиперпараметрами модели являются:
# - optimizer - метод, определяющий способ обновления весов нейронной сети, в данном случае используется среднеквадратичное распространение корня.
# - loss function - мера того, насколько хорошо модель предсказывает значение
# - metrics (метрики) — позволяет отображать необходимые метрики для каждого этапа обучения

# In[11]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# ## 5. Обучение модели

# batch size - определяет размер пакета данных, передаваемого в модель для обучения, после которого модель обновляет веса. Возможна как передача всего обучающего пакета сразу, так и передача части данных. Малые значения параметров позволяют достичь быстрой сходимости нейросети, но требуют большего времени на передачу данных. При больших значениях модель быстро прекращает своё обучение, но само обучение может быть менее эффективным.
# 
# epochs - количество итераций обучения нейросети. Нейросеть обучалась 15 эпох (при бОльшем значении нейросеть требует больше времени на обучение, при меньшем - не успевает хорошо обучиться).

# In[12]:


model.fit(X_train, Y_train_big, batch_size=64, epochs=15)


# ## 6. Оценка модели на тестовой выборке

# **Преобразую predictions к тому же виду, что и Y_test**

# In[13]:


# Вывожу сырой predictions
predictions = model.predict(X_test)
predictions


# In[14]:


# Получаю массив точных значений predictions
predictions = np.argmax(predictions, axis=1)

# One-hot-encoding для массива значений predictions
predictions = tf.keras.utils.to_categorical(
    predictions, num_classes=None, dtype='float32'
)

# Вывожу изменённый predictions
predictions


# In[15]:


#преобразую predictions и Y_test_numbers в номера категорий
predictions_by_big = [np.argmax(i, axis=None, out=None) for i in predictions]
Y_test_numbers = [np.argmax(i, axis=None, out=None) for i in Y_test]


# **Precision, recall, f1-score для модели с широкими классами**

# In[16]:


# скопирую метрики для сравнения в пункте 8
metrics_big = classification_report(Y_test_numbers, predictions_by_big, target_names=names_big, output_dict=True)

print(classification_report(Y_test_numbers, predictions_by_big, target_names=names_big))


# ## 7. Повторение шагов 4, 5, 6 для модели с узкими классами

# In[17]:


# сохранение старой модели
model.save('CNN_CIFAR-10_coarse')

# очистка графа старой модели
tf.keras.backend.clear_session()


# **Создаю модель, последний слой которой будет делать разделение на 100 классов**

# In[18]:


def build_cnn_model():
    cnn_model = tf.keras.Sequential([
                                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
                                    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
                                    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(100, activation='softmax') 
    ])

    return cnn_model

model = build_cnn_model()

# Единичный вызов модели для её инициализации
model.predict(X_train[[0]])

# Вывод характеристик модели
print(model.summary())


# In[19]:


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# **Обучу эту модель на узких классах**

# Вследствие увеличения количества классов, нейросеть стала учиться значительно дольше. На пример, нейросеть с 20 выходными нейронами за 10 эпох успевала достичь показателей loss: 1.0986 - accuracy: 0.6514

# In[20]:


model.fit(X_train, Y_train_small, batch_size=64, epochs=15)


# **Обрабатываю prediction**

# In[21]:


# Получаю сырой prediction
predictions = model.predict(X_test)

# Получаю массив точных значений predictions
predictions = np.argmax(predictions, axis=1)

# One-hot-encoding для массива значений predictions
predictions = tf.keras.utils.to_categorical(
    predictions, num_classes=None, dtype='float32'
)

#преобразую predictions в номера категорий
predictions = [np.argmax(i, axis=None, out=None) for i in predictions]


# **Считаю метрики**

# In[22]:


# Словарь для преобразования узких категорий в широкие
dict_names = {}
dict_names.update(dict.fromkeys(['beaver', 'dolphin', 'otter', 'seal', 'whale'], 'aquatic mammals'))
dict_names.update(dict.fromkeys(['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'], 'fish'))
dict_names.update(dict.fromkeys(['orchid', 'poppy', 'rose', 'sunflower', 'tulip'], 'flowers'))
dict_names.update(dict.fromkeys(['bottle', 'bowl', 'can', 'cup', 'plate'], 'food containers'))
dict_names.update(dict.fromkeys(['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'], 'fruit and vegetables'))
dict_names.update(dict.fromkeys(['clock', 'keyboard', 'lamp', 'telephone', 'television'], 'household electrical devices'))
dict_names.update(dict.fromkeys(['bed', 'chair', 'couch', 'table', 'wardrobe'], 'household furniture'))
dict_names.update(dict.fromkeys(['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'], 'insects'))
dict_names.update(dict.fromkeys(['bear', 'leopard', 'lion', 'tiger', 'wolf'], 'large carnivores'))
dict_names.update(dict.fromkeys(['bridge', 'castle', 'house', 'road', 'skyscraper'], 'large man-made outdoor things'))
dict_names.update(dict.fromkeys(['cloud', 'forest', 'mountain', 'plain', 'sea'], 'large natural outdoor scenes'))
dict_names.update(dict.fromkeys(['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'], 'large omnivores and herbivores'))
dict_names.update(dict.fromkeys(['fox', 'porcupine', 'possum', 'raccoon', 'skunk'], 'medium-sized mammals'))
dict_names.update(dict.fromkeys(['crab', 'lobster', 'snail', 'spider', 'worm'], 'non-insect invertebrates'))
dict_names.update(dict.fromkeys(['baby', 'boy', 'girl', 'man', 'woman'], 'people'))
dict_names.update(dict.fromkeys(['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'], 'reptiles'))
dict_names.update(dict.fromkeys(['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'], 'small mammals'))
dict_names.update(dict.fromkeys(['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'], 'trees'))
dict_names.update(dict.fromkeys(['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'], 'vehicles 1'))
dict_names.update(dict.fromkeys(['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'], 'vehicles 2'))


# In[23]:


# Преобразую узкие категории в широкие пользуясь словарём
predictions_by_small = [names_big.index(dict_names[names_small[i]]) for i in predictions]


# **Precision, recall, f1-score для модели с узкими классами**

# In[24]:


# скопирую метрики для сравнения в пункте 8
metrics_small = classification_report(Y_test_numbers, predictions_by_small, target_names=names_big, output_dict=True)

print(classification_report(Y_test_numbers, predictions_by_small, target_names=names_big))


# In[25]:


# сохранение модели
model.save('CNN_CIFAR-10_fine')


# ## 8. Сравню метрики моделей и отвечу на оставшиеся вопросы

# In[26]:


recall_small = [metrics_small[i]["recall"] for i in names_big]
recall_big = [metrics_big[i]["recall"] for i in names_big]
precision_small = [metrics_small[i]["precision"] for i in names_big]
precision_big = [metrics_big[i]["precision"] for i in names_big]
f1_score_small = [metrics_small[i]["f1-score"] for i in names_big]
f1_score_big = [metrics_big[i]["f1-score"] for i in names_big]


# In[27]:


n=len(names_big)
r = np.arange(n) 
width = 0.3
  
plt.bar(r, recall_small, color = 'b', 
        width = width, edgecolor = 'black', 
        label='by_small') 
plt.bar(r + width, recall_big, color = 'g', 
        width = width, edgecolor = 'black', 
        label='by_big') 

plt.xticks(rotation = 90)

plt.xlabel("Category name") 
plt.ylabel("Recall") 
plt.title("Recall comparison") 
  
plt.xticks(r+ width/2,[i for i in names_big]) 
plt.legend() 
  
plt.show()


# In[29]:


n=len(names_big)
r = np.arange(n) 
width = 0.3
  
plt.bar(r, precision_small, color = 'b', 
        width = width, edgecolor = 'black', 
        label='by_small') 
plt.bar(r + width, precision_big, color = 'g', 
        width = width, edgecolor = 'black', 
        label='by_big') 

plt.xticks(rotation = 90)

plt.xlabel("Category name") 
plt.ylabel("Precision") 
plt.title("Precision comparison") 
  
plt.xticks(r+ width/2,[i for i in names_big]) 
plt.legend() 
  
plt.show()


# In[30]:


n=len(names_big)
r = np.arange(n) 
width = 0.3
  
plt.bar(r, f1_score_small, color = 'b', 
        width = width, edgecolor = 'black', 
        label='by_small') 
plt.bar(r + width, f1_score_big, color = 'g', 
        width = width, edgecolor = 'black', 
        label='by_big') 

plt.xticks(rotation = 90)

plt.xlabel("Category name") 
plt.ylabel("f1-score") 
plt.title("f1-score comparison") 
  
plt.xticks(r+ width/2,[i for i in names_big]) 
plt.legend() 
  
plt.show()


# Графики, получившиеся для "Recall" и "Precision" получились абсолютно разными. На одном наибольшую эффективность показывает нейросеть, обучавшаяся на узких категориях, на другом наоборот. Чтобы разрешить эту неопределённость буду анализировать график f1-score, он уравнивает значимость "Recall" и "Precision". 
# 
# На графике f1-score видно, что модель с бОльшими классами показывает лучший результат, хотя в общей сложности итоги работы моделей схожи.
# 
# Рассмотрю классы, имеющие наибольшую разницу в результате. Думаю эта разница вызвана тем, что обьекты некоторых больших категорий слишком разнообразны (разные цвета, размер, форма, положение в кадре), поэтому нейросеть изучающая большие категории не может уловить тренд (такие категорий: "household electrical devices", "vehicles 1", "vehicles 2", "people"). Объекты же других категорий более однородны (к таким категориям можно отнести: "large man-made outdoor things", "large natural outdoor scenes", "household furniture", "food containers", "fish")

# In[ ]:




