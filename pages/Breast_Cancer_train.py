import pickle
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

def train_and_save():
    #Tải dữ liệu
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    #Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit =(X, y)

    #Đóng gói các thành phần cần thiết
    payload = {
        'model': model,
        'target_names': data.target_names,
        'feature': data.feature_names,
        'stats': X.describe()
    }

    with open('cancer_model.pkl', 'wb') as f:
        pickle.dump(payload, f)

    print('Đã lưu mô hình thành công file cancel_model.pkl')

if __name__ == "__main__":
    train_and_save()

    