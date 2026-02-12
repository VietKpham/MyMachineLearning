import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Breast Cancer ML", layout='wide')

#Tải mô hình đã đóng gói
@st.cache_resource
def load_model():
    with open('cancer_model.pkl', 'rb') as f:
        return pickle.load(f)
    
data_bundle = load_model()
model = data_bundle['model']
target_names = data_bundle['target_names']
feature_names = data_bundle['feature_names']
stats = data_bundle['stats']

#Giao diện SideBar nhập liệu
st.sidebar.header("Nhập chỉ số sinh thiết")
def get_user_input():
    user_data = {}
    #Chọn 5 chỉ số mặc định quan trọng
    main_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

    for name in main_features:
        if name in main_features:
            user_data[name] = st.sidebar.slider(
            name.capitalize(),
            float(stats.loc['min', name]),
            float(stats.loc['max', name]),
            float(stats.loc['mean', name])
            )
        else:
            user_data[name] = float(stats.loc['mean', name])
    return pd.DataFrame(user_data, index=[0])

input_df = get_user_input()


#Hiển thị kết quả dự đoán
st.title("Chuẩn đoán Ung thu vú")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('Dự đoán của ML')
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)

    res_text = target_names[prediction[0]].upper()
    confidence = np.max(proba) * 100

    if res_text == 'MALIGNANT':
        st.error(f"KẾT QUẢ: **{res_text}** (Ác tính)")
    else:
        st.success(f"KẾT QUẢ: **{res_text}** (Lành tính)")

    st.metric("Độ tin tưởng", f"{confidence:.2f}%")

#Biểu đồ nhiệt

with col2:
    st.subheader("So sánh với mức trung bình")

    compare_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

    #Tạo DataFrame
    user_vals = input_df[compare_features].iloc[0]
    avg_vals = stats.loc['mean', compare_features]

    compare_df = pd.DataFrame({
        'Chỉ số': compare_features,
        'Của bạn': user_vals.values,
        'Trung bình': avg_vals.values
    }).set_index('Chỉ số')

    fig, ax = plt.subplot()
    sns.heatmap(compare_df.T, annot=True, cmap="RdYlGn_r", fmt=".1f", ax=ax)
    st.pyplot(fig)

st.info("Các chỉ số cao hơn mức trung bình (vùng màu đỏ) thường là dấu hiệu U ác tính")