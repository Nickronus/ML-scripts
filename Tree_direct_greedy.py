import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def cross_validation(df, target_col, group_col) -> float:
    X = df.drop([target_col, group_col], axis=1)
    y = df[target_col]
    groups = df[group_col]

    model = DecisionTreeClassifier(random_state=42)
    group_kfold = GroupKFold(n_splits=10)
    scores = cross_val_score(model, X, y, groups=groups, cv=group_kfold, scoring='accuracy')
    
    return scores.mean()

if __name__ == '__main__':
    filename = 'C:\\Users\\Hot\\Downloads\\123\\result(2).xlsx'
    target_column = 'IS SICK'
    name_column = 'NAME'

    # Загрузка данных
    try:
        df = pd.read_excel(filename)
        df = df.iloc[:, 1:]  # Удаляем первый столбец (если это индекс)
        print("Данные успешно загружены.")
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        exit()

    df = df.dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype == 'object' and col != name_column:
            df[col] = LabelEncoder().fit_transform(df[col])

    all_features = [col for col in df.columns if col not in [target_column, name_column]]
    selected_features = []
    best_accuracy = 0.0

    for _ in range(len(all_features)):
        current_best_accuracy = best_accuracy
        best_feature = None

        for feature in all_features:
            if feature in selected_features:
                continue

            trial_features = selected_features + [feature]
            trial_df = df[[target_column, name_column] + trial_features].copy()

            accuracy = cross_validation(trial_df, target_column, name_column)
            print(f"Тест с признаками {trial_features}: accuracy = {accuracy:.4f}")

            if accuracy > current_best_accuracy:
                current_best_accuracy = accuracy
                best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
            best_accuracy = current_best_accuracy
            print(f"Добавлен признак {best_feature}, текущая accuracy = {best_accuracy:.4f}")
        else:
            print("Точность не улучшается, завершаем отбор.")
            break

    print("\nИтоговый набор признаков:", selected_features)
    print(f"Лучшая точность: {best_accuracy:.4f}")