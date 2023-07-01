import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pickle import dump, load

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def open_table(url):
    df = pd.read_csv(url)
    return df

def split_table(df):
    X = df.drop(["charges"], axis=1)
    y = df['charges']
    return X, y

def make_train_and_test_matrix(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, test = True):
    categorical = ['sex', 'smoker', 'region']
    numeric_features = ["age", "bmi", "children"]
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical),
        ('scaling', MinMaxScaler(), numeric_features)
    ])
    X_train_transformed = column_transformer.fit_transform(X_train)
    if test:
        X_test_transformed = column_transformer.transform(X_test)
    lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
    lst.extend(numeric_features)
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=lst)
    if test:
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=lst)
    if test:
        return X_train_transformed, X_test_transformed
    else:
        return X_train_transformed

def fit_and_save_model(X_train_transformed, y_train, X_test_transformed, y_test, model_path ="data/model_weights.mw", test = True):

    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    if test:
        pred = model.predict(X_test_transformed)
    else:
        pred = model.predict(X_train_transformed)

    with open(model_path, "wb") as file:
        dump(model, file)

    if test:
        mae = mean_absolute_error(y_test, pred)
        return f"Средняя абсолютная ошибка равна: {round(mae, 2)}"
    else:
        mae = mean_absolute_error(y_train, pred)
        return f"Средняя абсолютная ошибка равна: {round(mae, 2)}"

def load_model_and_predict(df, y_train, model_path="data/model_weights.mw"):
    with open(model_path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    mae = mean_absolute_error(y_train, prediction)

    if prediction > 0:
        return prediction, mae
    else:
        return 0, mae

if __name__ == "__main__":
    df = open_table(url)
    X, y = split_table(df)
    X_train, X_test, y_train, y_test = make_train_and_test_matrix(X, y)
    X_train_transformed, X_test_transformed = scale_data(X_train, X_test)
    fit_and_save_model(X_train_transformed, y_train, X_test_transformed, y_test, model_path="data/model_weights.mw")
    prediction, mae = load_model_and_predict(X_train_transformed, y_train, model_path="data/model_weights.mw")