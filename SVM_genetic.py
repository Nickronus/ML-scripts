import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deap import base, creator, tools, algorithms


def load_and_preprocess_data(filename, target_column, name_column):
    """Загружает данные, выполняет предварительную обработку, возвращает DataFrame."""
    try:
        dfr = pd.read_excel(filename)
        dfr = dfr.iloc[:, 1:]
        print("Данные успешно загружены из файла:", filename)
        print("Размер данных:", dfr.shape)
        print("Первые 5 строк данных:\n", dfr.head())
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден.")
        exit()
    except Exception as e:
        print(f"Ошибка при чтении файла '{filename}': {e}")
        exit()

    dfr = dfr.dropna()

    if name_column not in dfr.columns:
        print(f"Ошибка: Столбец с именами '{name_column}' не найден в файле.")
        exit()

    random_indices = np.random.permutation(dfr.index)
    dfr = dfr.loc[random_indices].reset_index(drop=True)

    for column in dfr.columns:
        if dfr[column].dtype == 'object':
            le = LabelEncoder()
            dfr[column] = le.fit_transform(dfr[column])

    return dfr

def cross_validation(df, target_column, name_column):
    """Выполняет кросс-валидацию с GroupKFold и SVM."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    groups = df[name_column]


    # Стандартизация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Создание модели SVM
    model_svc = SVC()

    # Кросс-валидация
    group_kfold = GroupKFold(n_splits=5) #  n_splits = 5 для скорости
    svc_scores = cross_val_score(model_svc, X, y, groups=groups, cv=group_kfold, scoring='accuracy')

    return svc_scores.mean()

def evaluate(individual, df, target_column, name_column, features):
    """Оценивает качество индивидуума (выбранных характеристик) с GroupKFold."""
    selected_features_indices = [i for i, bit in enumerate(individual) if bit == 1]
    selected_features = [features[i] for i in selected_features_indices]

    if not selected_features:
        return -1.0,

    try:
        df_selected = df[[target_column, name_column] + selected_features]
        cross_validation_result = cross_validation(df_selected, target_column, name_column)
        return cross_validation_result,
    except Exception as e:
        print(f"Ошибка при оценке: {e}")
        return -1.0,


def genetic_feature_selection(df, target_column, name_column, features, model, pop_size=50, cxpb=0.7, mutpb=0.2, ngen=100):
    """Выполняет генетический отбор признаков."""


    toolbox = base.Toolbox()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox.register("attribute", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attribute, n=len(features))

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda ind: evaluate(ind, df, target_column, name_column, features))

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                        cxpb=cxpb, mutpb=mutpb,
                                        ngen=ngen, stats=stats,
                                        halloffame=hof, verbose=True)

    best_individual = hof[0]
    selected_features_indices = [i for i, bit in enumerate(best_individual) if bit == 1] # Индексы
    selected_features = [features[i] for i in selected_features_indices] # Названия

    print("Лучший индивидуум:", best_individual)
    print("Фитнес лучшего индивидуума:", best_individual.fitness.values[0])
    print("Выбранные характеристики (индексы):", selected_features_indices)
    print("Выбранные характеристики (названия):", selected_features)

    return selected_features, best_individual.fitness.values[0]


if __name__ == '__main__':
    filename = 'C:\\Users\\Hot\\Downloads\\123\\result(2).xlsx'  # Замените на путь к вашему файлу
    target_column = 'IS SICK'
    name_column = 'NAME'

    # Загрузка и предварительная обработка данных
    dfr = load_and_preprocess_data(filename, target_column, name_column)

    # Список признаков
    features = [col for col in dfr.columns if col not in [target_column, name_column]]

    # Выбор модели (можно заменить на другую модель)
    model_svc = SVC()

    # Запуск генетического алгоритма
    selected_features, best_accuracy = genetic_feature_selection(dfr, target_column, name_column, features, model_svc)

    print(f"\nГенетический алгоритм завершен.")
    print(f"Лучшая точность: {best_accuracy:.4f}")
    print(f"Выбранные характеристики: {selected_features}")
