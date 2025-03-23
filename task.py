import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from kmodes.kmodes import KModes
import random

N = 20000

age_choices = ["18-25", "26-35", "36-45", "46+"]
gender_choices = ["Мужской", "Женский", "Предпочитаю не указывать"]
flight_freq_choices = ["Редко", "1–2 раза в год", "3–5 раз в год", "Часто"]
flight_type_choices = ["Короткий", "Средний", "Длительный"]
trip_purpose_choices = ["Работа", "Отдых", "Семья", "Другое"]
flight_class_choices = ["Эконом", "Бизнес", "Первый класс"]
activity_choices = ["Спать", "Читать", "Смотреть фильмы", "Работать"]
noise_choices = ["Терпимо", "Нейтрально", "Раздражает"]
neighbors_comm_choices = ["Люблю общаться", "Нейтрально", "Предпочитаю тишину"]
neighbor_type_pref_choices = ["Тихого", "Общительного", "Нейтрального"]
seat_choice_importance_choices = ["Очень важен", "Неважен"]
personal_space_choices = ["Очень важно", "Нейтрально", "Неважно"]
sleep_freq_choices = ["Всегда", "Часто", "Иногда", "Никогда"]
talk_to_strangers_choices = ["Положительно", "Нейтрально", "Отрицательно"]
extra_comfort_choices = ["Подушки", "Пледы", "Маски для сна"]

random_data = {
    "Возраст": [random.choice(age_choices) for _ in range(N)],
    "Пол": [random.choice(gender_choices) for _ in range(N)],
    "Частота полётов": [random.choice(flight_freq_choices) for _ in range(N)],
    "Тип перелёта": [random.choice(flight_type_choices) for _ in range(N)],
    "Цель поездки": [random.choice(trip_purpose_choices) for _ in range(N)],
    "Класс перелёта": [random.choice(flight_class_choices) for _ in range(N)],
    "Занятие в полёте": [random.choice(activity_choices) for _ in range(N)],
    "Отношение к шуму": [random.choice(noise_choices) for _ in range(N)],
    "Общение с соседями": [random.choice(neighbors_comm_choices) for _ in range(N)],
    "Предпочтение по типу соседа": [random.choice(neighbor_type_pref_choices) for _ in range(N)],
    "Выбор места": [random.choice(seat_choice_importance_choices) for _ in range(N)],
    "Личное пространство": [random.choice(personal_space_choices) for _ in range(N)],
    "Частота сна": [random.choice(sleep_freq_choices) for _ in range(N)],
    "Разговоры с незнакомцами": [random.choice(talk_to_strangers_choices) for _ in range(N)],
    "Дополнительные удобства": [random.choice(extra_comfort_choices) for _ in range(N)],
}

df = pd.DataFrame(random_data)


allowed_values = {
    "Возраст": {"18-25", "26-35", "36-45", "46+"},
    "Пол": {"Мужской", "Женский", "Предпочитаю не указывать"},
    "Частота полётов": {"Редко", "1–2 раза в год", "3–5 раз в год", "Часто"},
    "Тип перелёта": {"Короткий", "Средний", "Длительный"},
    "Цель поездки": {"Работа", "Отдых", "Семья", "Другое"},
    "Класс перелёта": {"Эконом", "Бизнес", "Первый класс"},
    "Занятие в полёте": {"Спать", "Читать", "Смотреть фильмы", "Работать"},
    "Отношение к шуму": {"Терпимо", "Нейтрально", "Раздражает"},
    "Общение с соседями": {"Люблю общаться", "Нейтрально", "Предпочитаю тишину"},
    "Предпочтение по типу соседа": {"Тихого", "Общительного", "Нейтрального"},
    "Выбор места": {"Очень важен", "Неважен"},
    "Личное пространство": {"Очень важно", "Нейтрально", "Неважно"},
    "Частота сна": {"Всегда", "Часто", "Иногда", "Никогда"},
    "Разговоры с незнакомцами": {"Положительно", "Нейтрально", "Отрицательно"},
    "Дополнительные удобства": {"Подушки", "Пледы", "Маски для сна"},
}


def clean_value(value, allowed_set):
    return value if value in allowed_set else np.nan


for col, allowed_set in allowed_values.items():
    df[col] = df[col].apply(lambda x: clean_value(x, allowed_set))


n_clusters = 6

kmodes = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=1)

clusters = kmodes.fit_predict(df.fillna('NULL'))

df['Cluster'] = clusters

print("Распределение по кластерам:")
print(df['Cluster'].value_counts())

# Предположим, что у нас 200 пассажиров, и мы хотим сравнить "стандартную рассадку" и "персонализированную".
# Сгенерируем случайные значения "удовлетворенности" для каждой группы.

# n_passengers = 200
#
# # Стандартная рассадка
# control_group = np.random.normal(loc=6.5, scale=1.0, size=n_passengers // 2)
# # Персонализированная рассадка
# experiment_group = np.random.normal(loc=7.0, scale=1.0, size=n_passengers // 2)
#
# data_test = pd.DataFrame({
#     'group': ['control'] * (n_passengers // 2) + ['experiment'] * (n_passengers // 2),
#     'satisfaction': np.concatenate([control_group, experiment_group])
# })
#
# print("\nСтатистика по группам:")
# print(data_test.groupby('group')['satisfaction'].describe())
#
# # Проведение t-теста
# t_stat, p_value = stats.ttest_ind(control_group, experiment_group)
# print("\nРезультаты t-теста:")
# print("t-статистика:", t_stat)
# print("p-value:", p_value)

