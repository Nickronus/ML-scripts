import pandas as pd
from sklearn.model_selection import cross_val_score, GroupKFold
import openpyxl
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def cross_validation(df) -> float:
    # Разделение на признаки (X) и целевую переменную (y)
    if target_column not in df.columns:
        print(f"Ошибка: Целевой столбец '{target_column}' не найден в файле.")
        exit()

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    groups = df[name_column]

    # # Кросс-валидация
    group_kfold = GroupKFold(n_splits=10)

    # Стандартизация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Создание модели SVM
    model_svc = SVC()

    # Кросс-валидация
    svc_cores = cross_val_score(model_svc, X, y, groups=groups, cv=group_kfold, scoring='accuracy')

    return svc_cores.mean()


if __name__ == '__main__':
    filename = 'C:\\Users\\Hot\\Downloads\\123\\result(2).xlsx'
    #filename = input('Введите путь до файла: ')
    target_column = 'IS SICK'
    name_column = 'NAME'

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

    # Удаляем строки с пропущенными значениями.
    dfr = dfr.dropna()

    # Проверяем, есть ли столбец с именами
    if name_column not in dfr.columns:
        print(f"Ошибка: Столбец с именами '{name_column}' не найден в файле.")
        exit()

    # Перемешиваем строки
    random_indices = np.random.permutation(dfr.index)
    dfr = dfr.loc[random_indices].reset_index(drop=True)

    # Преобразуем категориальные признаки в числовые, если они есть.
    for column in dfr.columns:
        if dfr[column].dtype == 'object':
            le = LabelEncoder()
            dfr[column] = le.fit_transform(dfr[column])


    dfr2 = dfr.copy()

    # Разделение на признаки (X) и целевую переменную (y)
    if target_column not in dfr2.columns:
        print(f"Ошибка: Целевой столбец '{target_column}' не найден в файле.")
        exit()

    X = dfr2.drop(target_column, axis=1)
    y = dfr2[target_column]
    groups = dfr2[name_column]

    # # Создание модели дерва решений
    # model_tree = DecisionTreeClassifier()

    # # Кросс-валидация
    group_kfold = GroupKFold(n_splits=10)

    # Стандартизация признаков
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Создание модели SVM
    model_svc = SVC()

    # Кросс-валидация
    svc_cores = cross_val_score(model_svc, X, y, groups=groups, cv=group_kfold, scoring='accuracy')

    svc_average_accuracy = svc_cores.mean()
    print(svc_average_accuracy)

    features = [
        "F2",
        "F1",
        "SHIMMER_LOCAL",
        "F0_MIN",
        "F4",
        "F0_MEAN",
        "F3",
        "JITTER_PPQ5",
        "F0_RANGE",
        "F0_MAX",
        "F0_STDEV",
        "DURATION",
        "JITTER_LOCAL",
        "HNR",
        "INTENSITY_RANGE",
        "INTENSITY_STDEV",
        "INTENSITY_MEAN"
    ]

    deleted = []
    df_result = dfr.copy()
    
    for i in range(16):
        result = {}
        for feature in features:
            df = df_result.copy()
            if feature in df.columns: # Добавляем проверку существования колонки
                df = df.drop(feature, axis=1) # Предполагается, что df - ваш DataFrame
            else:
                print(f"Признак {feature} не найден в DataFrame.")

            cross_validation_result = cross_validation(df)
            print(cross_validation_result)
            if svc_average_accuracy <= cross_validation_result:
                result[feature] = cross_validation_result - svc_average_accuracy
            
        print(result)

        value_of_item_for_delete = 0
        item_for_delete = None

        if len(result) == 0:
            break

        for key, value in result.items():
            if value > value_of_item_for_delete:
                value_of_item_for_delete = value
                item_for_delete = key

        df_result = df_result.drop(item_for_delete, axis=1)
        print('Удалён ', item_for_delete)
        svc_average_accuracy = svc_average_accuracy + value_of_item_for_delete
        print('Результат ', svc_average_accuracy)
        deleted.append(item_for_delete)
        features.remove(item_for_delete)

    print(deleted)
    print(svc_average_accuracy)

