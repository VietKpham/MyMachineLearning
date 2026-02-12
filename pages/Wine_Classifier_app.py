import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#Cấu hình trang
st.set_page_config(page_title="Wine Classifier", layout="wide")

# Tải dữ liệu và huấn luyện

@st.cache_resource
def train_model():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    #Huấn luyện model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model, wine.target_names, X.describe()

model, target_names, stats = train_model()

st.sidebar.header("Nhập thông số hóa học")

def user_input_features():
    inputs = {}
    #Tạo thanh trượt dựa trên Min/Max của bộ dữ liệu gốc
    for col in stats.columns:
        inputs[col] = st.sidebar.slider(
            col.capitalize(),
            float(stats.loc['min', col]),
            float(stats.loc['max', col]),
            float(stats.loc['mean', col])
        )
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

st.title("Dự đoán loại rượu")
st.write('Ứng dụng sử dụng mô hình Random Forest để phân loại rượu')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông số bạn đã chọn: ")
    st.write(input_df.T) 

with col2:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Kết quả dự đoán: ")
    st.success(f"Loại rượu dự đoán: **{target_names[prediction[0]]}**") 

    st.subheader("Xác suất dự đoán (%):")
    proba_df = pd.DataFrame(prediction_proba, columns=target_names)
    st.bar_chart(proba_df.T)
           

