import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Загрузка данных
data_spam = pd.read_excel('messages_spam.xlsx')
data_spam['label'] = 1
data_no_spam = pd.read_excel('messages_nospam.xlsx')
data_no_spam['label'] = 0

df = pd.concat([data_spam, data_no_spam], ignore_index=True)

# Очистка текста
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Удаление специальных символов
    text = text.lower()  # Приведение к нижнему регистру
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Удаление стоп-слов
    return ' '.join(tokens)

df['Cleaned_Message'] = df['Message'].apply(clean_text)
df.drop_duplicates(subset=['Cleaned_Message'], inplace=True)

# Векторизация с использованием TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned_Message']).toarray()
y = df['label']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение модели
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),# Увеличено количество нейронов
    layers.Dropout(0.5),  # Регуляризация
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model_history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Вывод метрик
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Визуализация потерь и точности
plt.plot(model_history.history['loss'], label='loss')
plt.plot(model_history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.ylabel('Value')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.plot(model_history.history['accuracy'], label='accuracy')
plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.ylabel('Value')
plt.xlabel('Epochs')
plt.legend()
plt.show()
