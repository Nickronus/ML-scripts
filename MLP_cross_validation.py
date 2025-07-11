import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

def load_and_preprocess_data(filename, target_column, name_column, random_state=42):
    """Загружает и предобрабатывает данные с групповым перемешиванием."""
    try:
        df = pd.read_excel(filename)
        df = df.iloc[:, 1:]
        print("Данные успешно загружены из файла:", filename)
        print("Исходный размер данных:", df.shape)
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        exit()

    df = df.dropna()
    print(f"Размер после удаления пропусков: {df.shape}")

    if name_column not in df.columns:
        print(f"Ошибка: Столбец '{name_column}' не найден.")
        exit()

    unique_patients = df[name_column].unique()
    np.random.seed(random_state)
    shuffled_patients = np.random.permutation(unique_patients)
    
    shuffled_df = pd.concat([df[df[name_column] == patient] for patient in shuffled_patients])
    shuffled_df = shuffled_df.reset_index(drop=True)
    
    for column in shuffled_df.columns:
        if shuffled_df[column].dtype == 'object' and column != name_column:
            le = LabelEncoder()
            shuffled_df[column] = le.fit_transform(shuffled_df[column].astype(str))

    return shuffled_df

def ten_fold_cross_validation(df, target_column, name_column, model, random_state=42):
    selected_features = [
        "F2", "F1", "SHIMMER_LOCAL", "F0_MIN", "F4", "F0_MEAN", "F3",
        "JITTER_PPQ5", "F0_RANGE", "F0_MAX", "F0_STDEV", "DURATION",
        "JITTER_LOCAL", "HNR", "INTENSITY_RANGE", "INTENSITY_STDEV", "INTENSITY_MEAN"
    ]
    
    X = df[selected_features]
    y = df[target_column]
    groups = df[name_column]
    
    group_kfold = GroupKFold(n_splits=10)
    metrics = {
        'accuracy': [],
        'roc_auc': [],
        'f1_score': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1] if hasattr(fold_model, "predict_proba") else None
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['f1_score'].append(f1_score(y_test, y_pred))
        
        if y_proba is not None and len(np.unique(y_test)) >= 2:
            metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        else:
            metrics['roc_auc'].append(np.nan)
        
        print(f"\nFold {fold}:")
        print(f"  Patients in train: {len(np.unique(groups.iloc[train_idx]))}")
        print(f"  Patients in test: {len(np.unique(groups.iloc[test_idx]))}")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
        print(f"  F1 Score: {metrics['f1_score'][-1]:.4f}")
        if not np.isnan(metrics['roc_auc'][-1]):
            print(f"  ROC AUC: {metrics['roc_auc'][-1]:.4f}")

    print("\nИтоговые метрики:")
    print(f"Accuracy: {np.nanmean(metrics['accuracy']):.4f} ± {np.nanstd(metrics['accuracy']):.4f}")
    print(f"ROC AUC: {np.nanmean(metrics['roc_auc']):.4f} ± {np.nanstd(metrics['roc_auc']):.4f}")
    print(f"F1 Score: {np.nanmean(metrics['f1_score']):.4f} ± {np.nanstd(metrics['f1_score']):.4f}")
    
    return metrics

if __name__ == '__main__':
    filename = 'C:\\Users\\Hot\\Downloads\\123\\result(2).xlsx'
    target_column = 'IS SICK'
    name_column = 'NAME'
    random_state = 42
    
    df = load_and_preprocess_data(filename, target_column, name_column, random_state)
    
    best_hidden_layer_sizes = (100,)
    best_activation = 'tanh'
    best_solver = 'lbfgs'
    best_alpha = 0.002776688478137217
    best_learning_rate = 'invscaling'
    best_batch_size = 128

    model = MLPClassifier(
        hidden_layer_sizes=best_hidden_layer_sizes,
        activation=best_activation,
        solver=best_solver,
        alpha=best_alpha,
        learning_rate=best_learning_rate,
        batch_size=best_batch_size,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
        verbose=True,
        warm_start=False
    )
    
    metrics = ten_fold_cross_validation(df, target_column, name_column, model, random_state)