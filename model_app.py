import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
from model_file import url, open_table, split_table, scale_data, load_model_and_predict

st.set_page_config(
    layout="wide",
)

def show_title():
    image_up = Image.open("data/myriam-zilles-KltoLK6Mk-g-unsplash(up).jpg")
    st.image(image_up)
    st.title("Предсказание стоимости страховки на здоровье")
    st.header("на основе модели линейной регрессии")
    st.divider()

def main_page():
    col1, col2 = st.columns([0.3, 0.7], gap = "large")
    with col1:
        st.header("Введите данные")
        age = st.slider("Возраст", min_value=18, max_value=90, value=30,
                          step=1)
        sex = st.radio("Пол", ("Мужской", "Женский"))
        bmi = st.number_input("Индекс массы тела:", value = 25)
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
        train_X_df, train_y_df = split_table(train_df)
        full_X_df = pd.concat((user_data, train_X_df), axis=0)
        preprocessed_X_df = scale_data(full_X_df, [], test=False)
        st.subheader("Предсказание")
        prediction = load_model_and_predict(preprocessed_X_df)
        if prediction > 0:
            st.info(f"Предсказанная стоимость страховки: {prediction}")
        else:
            st.info(f"Предсказанная стоимость страховки: {prediction}. Вам страховка не нужна:)")



    with col2:
        st.header("Что влияет на стоимость страховки больше всего?")
        tab1, tab2, tab3 = st.tabs(["Курение", "Индекс массы тела", "Возраст"])

        with tab1:
            fig1 = px.box(data_frame = train_df, x = "smoker", y = "charges", points="all")
            st.plotly_chart(fig1)
            st.markdown("Курение оказывает значительное влияние на стоимость страховки. Наличие факта курения повышает страховку на более чем **23.000** условные единицы. Поэтому если вы хотите сократить стоимость страховки на здоровье, вам нужно бросить курение в первую очередь.")
        with tab2:
            fig2 = px.scatter(data_frame=train_df, x = "bmi", y = "charges",  color = "smoker", color_discrete_sequence= px.colors.qualitative.Bold)
            st.plotly_chart(fig2)
            st.write("При увеличении индекса массы тела стоимость страховки на здоровье повышается. После отказа от курения вам нужно сократить значение индекса - похудеть.")
        with tab3:
            fig3 = px.scatter(data_frame= train_df, x = "age", y = "charges", color = "bmi", color_continuous_scale= px.colors.sequential.Plasma)
            st.plotly_chart(fig3)
            st.write("В больших возрастах стоимость страховки увеличивается - это необходимо учитывать при покупке страховки на здоровье.")

def additional_info():
    st.divider()
    st.write("Использованный набор данных был взят из репозитория по ссылке: https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv")
    image_down = Image.open("data/myriam-zilles-KltoLK6Mk-g-unsplash(down).jpg")
    st.image(image_down)

def process_main_page():
    show_title()
    main_page()
    additional_info()

if __name__ == "__main__":
    process_main_page()


