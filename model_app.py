import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
from model_file import url, open_table, split_table, scale_data, load_model_and_predict

st.set_page_config(
    page_icon=Image.open("images/pills.png"),
    layout="wide",
)


def show_title():
    image_up = Image.open("images/myriam-zilles-KltoLK6Mk-g-unsplash(up).jpg")
    st.image(image_up)
    st.title("Prediction insurance cost")
    st.header("based on Linear regression model")
    st.divider()


def main_page():
    col1, col2 = st.columns([0.3, 0.7], gap="large")
    with col1:
        st.header("Input data")
        age = st.slider("Age", min_value=18, max_value=90, value=30,
                        step=1)
        sex = st.radio("Gender", ("Male", "Female"))
        bmi = st.number_input("Body mass index:", value=25)
        children = st.slider("Children", min_value=0, max_value=5, value=1,
                             step=1)
        smoker = st.radio("Smoker", ("Yes", "No"))
        region = st.selectbox("Region", ("northwest", "southeast", "northeast", "southwest"))
        translate = {"Male": "male", "Female": "female", "Yes": "yes", "No": "no"}

        data = {
            "age": age,
            "sex": translate[sex],
            "bmi": bmi,
            "children": children,
            "smoker": translate[smoker],
            "region": region
        }

        user_data = pd.DataFrame(data, index=[0])
        st.subheader("Your data")
        st.write(user_data)

        train_df = open_table(url)
        train_X_df, train_y_df = split_table(train_df)
        full_X_df = pd.concat((user_data, train_X_df), axis=0)
        preprocessed_X_df = scale_data(full_X_df, [], test=False)
        st.header("Prediction")
        prediction = load_model_and_predict(preprocessed_X_df)
        if prediction > 0:
            st.markdown(f"Predicted insurance cost: **{prediction}**")
        else:
            st.info(f"Predicted insurance cost: **{prediction}**. You don't need an insurance:)")

    with col2:
        st.header("What cause such insurance cost?")
        tab1, tab2, tab3 = st.tabs(["Smoking", "Body mass index", "Age"])

        with tab1:
            fig1 = px.histogram(data_frame=train_df, x='smoker', y='charges', opacity=0.6, histfunc='avg', color='mediumslateblue')
            st.plotly_chart(fig1)
            st.markdown(
                "Smoking has a significant impact on the cost of insurance. The fact of smoking increases insurance "
                "by more than **23,000** units. So if you want to reduce the cost of your health insurance, "
                "you need to quit smoking first.")
        with tab2:
            fig2 = px.scatter(data_frame=train_df, x="bmi", y="charges", opacity=0.7)
            st.plotly_chart(fig2)
            st.write(
                "As your body mass index increases, the cost of health insurance increases too. After quitting "
                "smoking, you need to reduce the index value - lose weight.")
        with tab3:
            fig3 = px.scatter(data_frame=train_df, x="age", y="charges", opacity=0.7)
            st.plotly_chart(fig3)
            st.write("At older ages, the cost of insurance increases - this must be taken into account when "
                     "purchasing health insurance.")


def additional_info():
    st.divider()
    st.write(
        "The data was retrived from the repository via the [link]("
        "https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv).")
    image_down = Image.open("images/myriam-zilles-KltoLK6Mk-g-unsplash(down).jpg")
    st.image(image_down)


def process_main_page():
    show_title()
    main_page()
    additional_info()


if __name__ == "__main__":
    process_main_page()
