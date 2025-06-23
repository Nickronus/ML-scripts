import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from deap import base, creator, tools, algorithms

def load_and_preprocess_data(filename, target_column, name_column):
    """Загружает и предварительно обрабатывает данные"""
    try:
        df = pd.read_excel(filename)
        df = df.iloc[:, 1:]
        print("Данные успешно загружены")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        exit()

    df = df.dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for col in df.columns:
        if df[col].dtype == 'object' and col != name_column:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    return df

def evaluate(individual, df, target_col, group_col, features):
    """Функция оценки для генетического алгоритма"""
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    selected_features = [features[i] for i in selected_indices]
    
    if not selected_features:
        return -1.0,
    
    try:
        trial_df = df[[target_col, group_col] + selected_features].copy()

        X = trial_df.drop([target_col, group_col], axis=1)
        y = trial_df[target_col]
        groups = trial_df[group_col]

        model = DecisionTreeClassifier(random_state=42)
        cv = GroupKFold(n_splits=5)
        scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='accuracy')
        
        return scores.mean(),
    except Exception as e:
        print(f"Ошибка оценки: {e}")
        return -1.0,

def genetic_feature_selection(df, target_col, group_col, features, 
                             pop_size=30, cxpb=0.7, mutpb=0.2, ngen=15):
    """Генетический алгоритм для отбора признаков"""

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_bool, n=len(features))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", lambda ind: evaluate(ind, df, target_col, group_col, features))
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
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, 
                                  ngen=ngen, stats=stats, halloffame=hof, verbose=True)
    
    best_ind = hof[0]
    selected = [features[i] for i, val in enumerate(best_ind) if val == 1]
    best_score = best_ind.fitness.values[0]
    
    return selected, best_score

if __name__ == '__main__':
    filename = 'C:\\Users\\Hot\\Downloads\\123\\result(2).xlsx'
    target_column = 'IS SICK'
    name_column = 'NAME'
    
    # Загрузка данных
    df = load_and_preprocess_data(filename, target_column, name_column)
    
    # Список признаков для отбора
    features = [col for col in df.columns if col not in [target_column, name_column]]
    print(f"\nДоступные признаки для отбора ({len(features)}): {features}")
    
    # Запуск генетического алгоритма
    selected_features, best_score = genetic_feature_selection(
        df, target_column, name_column, features,
        pop_size=30, cxpb=0.7, mutpb=0.2, ngen=15
    )
    
    print("\nРезультаты:")
    print(f"Лучшая точность: {best_score:.4f}")
    print(f"Выбранные признаки ({len(selected_features)}): {selected_features}")