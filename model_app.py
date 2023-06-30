import pandas as pd
import streamlit as st
from model_file import url, open_table, split_table, scale_data, load_model_and_predict

def make_columns():
    col1, col2 = st.columns([0.3, 0.7])
    return col1, col2

def process_main_page():
    global col1
    show_title()
    process_inputs_col1(col1)

def show_title():
    st.title("Предсказание стоимости страховки")
    st.header("на основе модели линейной регрессии")
    st.divider()

def input_features(col1):
    col1.subheader("Введите данные")
    age = col1.slider("Возраст", min_value=18, max_value=90, value=18,
                            step=1)
    sex = col1.radio("Пол", ("Мужской", "Женский"))
    bmi = col1.number_input("Индекс массы тела:")
    children = col1.slider("Количество детей", min_value=0, max_value=5, value=1,
                         step=1)
    smoker = col1.radio("Являетесь ли вы курильщиком?", ("Да", "Нет"))
    region = col1.selectbox("Регион проживания", ("northwest", "southeast", "northeast", "southwest"))
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

def write_user_data(col1, df):
    col1.subheader("Ваши данные")
    col1.write(df)

def process_inputs_col1(col1):
    col1.subheader('Заданные пользователем параметры')
    user_input_df = input_features(col1)
    write_user_data(col1, user_input_df)
    train_df = open_table(url)
    train_X_df, _ = split_table(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = scale_data(full_X_df, [], test = False)
    col1.subheader("Предсказание")
    col1.write(f"Предсказанная стоимость страховки: {load_model_and_predict(preprocessed_X_df)}")

if __name__ == "__main__":
    process_main_page()


