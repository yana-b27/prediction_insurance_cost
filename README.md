# prediction_insurance_cost

![Medicine](https://github.com/yana-b27/prediction_insurance_cost/blob/main/images/myriam-zilles-KltoLK6Mk-g-unsplash(up).jpg)

This project is about a prediction of health insurance cost using Linear regression model and creating an application for prediction the cost through Python porgramming language and Streamlit framework. 

## Dataset info

The dataset from the book "Machine Learning with R" was downloaded via the [link](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv). There are the next features in the dataset:

- `age`: age of the respondent
- `sex`: gender of the respondent
- `bmi`: bosy mass index
- `children`: the amount of children
- `smoker`: the indicator if the respondent smokes
- `region`: the region where respondent lives
- `charges`: the cost of insurance cost

## Used Pyhton libraries

Exploratory data analysis was processed with pandas and seaborn libraries. To prepare data for building a model, the sklearn module was imported to encode categorical features, form a training and test set. Then, using the same library, Linear regresssion model was used for test data prediciton. Streamlit framework was used for creating a web application, with pillow library images were visualized in the app and plotly module was used for creating interactive graphs.

## Repository structure

- [pred_insurance.ipynb](https://github.com/yana-b27/prediction_insurance_cost/blob/main/eda/pred_insurance.ipynb) в папке eda_and_prediction_model - exploratory data analysis and building Linear regression model
- [model_file.py](https://github.com/yana-b27/prediction_insurance_cost/blob/main/model_file.py) -data processing and prediction model collected in Pyhton functions
- [model_app.py](https://github.com/yana-b27/prediction_insurance_cost/blob/main/model_app.py) - code of the app built on Streamlit framework, which imports functions of the model_file module
- [data](https://github.com/yana-b27/prediction_insurance_cost/tree/main/data) - folder with the coefficients of the model stored in the file model_weights.mw

Used images of the application were downloaded via the [link](https://unsplash.com/photos/KltoLK6Mk-g)

## Web app info

**The app is available via the [link](https://prediction-insurance-cost.streamlit.app/)**

In streamlit appplication users can input features for prediction insurance cost in the special field. In parallel with the data entry field in the application, there is a field that interprets the resulting prediction result based on the model. It features interactive graphs that help the user interpret the prediction along with explanatory text.
