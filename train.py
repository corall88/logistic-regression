import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from data_preprocessing import data_prepare
from joblib import dump

MAX_ITERATIONS = 2500
FILE_NAME = 'model.joblib'

def train_logistic_regression(X_train: np.array, y_train: np.array, max_iter: int = MAX_ITERATIONS) -> LogisticRegression:
    """
    Обучает модель логистической регрессии на обучающем наборе данных.

    Parameters:
    X_train (np.array): Входные признаки для обучения.
    y_train (np.array): Целевые значения для обучения.
    max_iter (int): Максимальное количество итераций.

    Returns:
    LogisticRegression: Обученная модель логистической регрессии.
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def plot_roc_curve(y_test: np.array, y_prob: np.array) -> None:
    """
    Строит кривую ROC.

    Parameters:
    y_test (np.array): Истинные метки классов.
    y_prob (np.array): Вероятности принадлежности к классу.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_test: np.array, y_pred: np.array) -> None:
    """
    Строит матрицу ошибок.

    Parameters:
    y_test (np.array): Истинные метки классов.
    y_pred (np.array): Предсказанные метки классов.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Истинные метки')
    plt.ylabel('Предсказанные метки')
    plt.title('Матрица ошибок')
    plt.show()

def evaluate_model(model: LogisticRegression, X_test: np.array, y_test: np.array) -> None:
    """
    Выполняет оценку модели, выводит метрики и строит графики.

    Parameters:
    model (LogisticRegression): Обученная модель для оценки.
    X_test (np.array): Входные признаки для тестирования.
    y_test (np.array): Целевые значения для тестирования.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plot_confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def save_model(model: LogisticRegression, file_name: str) -> None:
    """
    Сохраняет обученную модель в файл.

    Parameters:
    model (LogisticRegression): Обученная модель для сохранения.
    file_name (str): Имя файла для сохранения модели.
    """
    dump(model, file_name)
    print(f"Модель успешно сохранена в файл: {file_name}")

def main() -> None:
    """
    Основная функция для подготовки данных, обучения модели, её оценки и сохранения в файл.
    """
    X_train, X_test, y_train, y_test = data_prepare()

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели на масштабированных данных
    model = train_logistic_regression(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)

    # Сохранение модели в файл
    save_model(model, FILE_NAME)

if __name__ == "__main__":
    main()
