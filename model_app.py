import pandas as pd
import streamlit as st
from model_file import url, open_table, split_table, scale_data, load_model_and_predict

def main_page():
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        st.subheader("Введите данные")
        age = st.slider("Возраст", min_value=18, max_value=90, value=18,
                          step=1)
        sex = st.radio("Пол", ("Мужской", "Женский"))
        bmi = st.number_input("Индекс массы тела:")
        children = st.slider("Количество детей", min_value=0, max_value=5, value=1,
                               step=1)
        smoker = st.radio("Являетесь ли вы курильщиком?", ("Да", "Нет"))
        region = st.selectbox("Регион проживания", ("northwest", "southeast", "northeast", "southwest"))
        translate = {"Мужской": "male", "Женский": "female", "Да": "yes", "Нет": "no"}

        data = {
            "age": age,
            "sex": translate[sex],
            "bmi": bmi,
            "children": children,
            "smoker": translate[smoker],
            "region": region
        }

        user_data = pd.DataFrame(data, index=[0])
        st.subheader("Ваши данные")
        st.write(user_data)

        train_df = open_table(url)
        train_X_df, _ = split_table(train_df)
        full_X_df = pd.concat((user_data, train_X_df), axis=0)
        preprocessed_X_df = scale_data(full_X_df, [], test=False)
        col1.subheader("Предсказание")
        col1.write(f"Предсказанная стоимость страховки: {load_model_and_predict(preprocessed_X_df)}")

def process_main_page():
    show_title()
    main_page()

def show_title():
    st.title("Предсказание стоимости страховки")
    st.header("на основе модели линейной регрессии")
    st.divider()

if __name__ == "__main__":
    process_main_page()


