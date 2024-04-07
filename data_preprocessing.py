from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def data_prepare():
    # Загружаем датасет
    data = load_breast_cancer()
    
    # Создаем DataFrame для удобной работы с данными
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Конвертируем метки из чисел в бинарные значения, где 0 соответствует 'B' (доброкачественная опухоль),
    # а 1 - 'M' (злокачественная опухоль). В данном случае этот шаг может быть пропущен,
    # так как метки уже в нужном формате.
    
    # Делим данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=92
    )
    
    # Преобразуем y_train и y_test в одномерные массивы для удобства использования в моделях
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    return X_train, X_test, y_train, y_test
