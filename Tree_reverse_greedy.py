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

    current_features = [col for col in df.columns if col not in [target_column, name_column]]
    removed_features = []
    best_accuracy = cross_validation(df, target_column, name_column)
    print(f"Начальная точность со всеми признаками: {best_accuracy:.4f}")

    while len(current_features) > 1:
        worst_feature = None
        current_worst_accuracy = best_accuracy

        for feature in current_features:
            trial_features = [f for f in current_features if f != feature]
            trial_df = df[[target_column, name_column] + trial_features].copy()
            
            accuracy = cross_validation(trial_df, target_column, name_column)
            print(f"Тест без признака {feature}: accuracy = {accuracy:.4f}")

            if accuracy >= current_worst_accuracy:
                current_worst_accuracy = accuracy
                worst_feature = feature

        if worst_feature is not None:
            current_features.remove(worst_feature)
            removed_features.append(worst_feature)
            best_accuracy = current_worst_accuracy
            print(f"Удален признак {worst_feature}, текущая accuracy = {best_accuracy:.4f}")
        else:
            print("Не удалось улучшить точность удалением признаков. Завершаем.")
            break

    print("\nОставшиеся признаки:", current_features)
    print(f"Удаленные признаки (в порядке удаления): {removed_features}")
    print(f"Лучшая достигнутая точность: {best_accuracy:.4f}")