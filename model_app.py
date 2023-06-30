import pandas as pd
import streamlit as st
from model_file import url, open_table, split_table, scale_data, load_model_and_predict

def process_main_page():
    show_title()
    process_inputs()

def show_title():
    st.header("Предсказание стоимости страховки")
    st.subheader("на основе модели линейной регрессии")
    st.divider()
    st.subheader("Введите данные")

def input_features():
    age = st.slider("Возраст", min_value=18, max_value=90, value=18,
                            step=1)
    sex = st.radio("Пол", ("Мужской", "Женский"))
    bmi = st.number_input("Индекс массы тела:")
    children = st.slider("Количество детей", min_value=0, max_value=5, value=1,
                         step=1)
    smoker = st.radio("Являетесь ли вы курильщиком?", ("Да", "Нет"))
    region = st.selectbox("Регион проживания", ("Northwest", "Southeast", "Northeast", "Southwest"))
    translate = {"Мужской": "male", "Женский": "female", "Да": "yes", "Нет": "no"}

    data = {
        "age": age,
        "sex": translate[sex],
        "bmi": bmi,
        "children": children,
        "smoker": translate[smoker],
        "region": region
    }

    user_data = pd.DataFrame(data, index = [0])
    return user_data

def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)

def write_prediction(prediction):
    st.write("## Предсказание")
    st.write(prediction)

def process_inputs():
    st.header('Заданные пользователем параметры')
    user_input_df = input_features()
    write_user_data(user_input_df)
    train_df = open_table(url)
    train_X_df, _ = split_table(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = scale_data(full_X_df, [], test = False)
    text_prediction = load_model_and_predict(preprocessed_X_df)
    write_prediction(text_prediction)

if __name__ == "__main__":
    process_main_page()


