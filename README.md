# prediction_insurance_cost

Данный проект посвящен предсказанию стоимости страховки на здоровье с помощью модели линейной регрессии и создания приложения для предсказания стоимости. В проекте использовался датасет из книги "Machine Learning with R", скачанный по [ссылке](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv). В наборе данных содержится следующая информация:

- возраст (поле age)
- пол (поле sex)
- индекс массы тела (поле bmi)
- кол-во детей у респондента (поле children)
- является ли респондент курильщиком или нет (поле smoker)
- регион проживания (поле region)
- стоимость страховки здоровья (поле charges)

Разведочный анализ данных произведен с помощью библиотек pandas и seaborn для построения графиков. Доя подготовки данных к построению модели использовался модуль sklearn для кодирования категориальных признаков, сформирования обучающей и тестовой выборки, в дальнейшем с этим модулем было произведено построение модели линейной регрессии. Для создания приложения применялась библиотека streamlit, pillow для загрузки изображений, plotly для создания интерактивных графиков, описывающих данные и объясняющих полученную стоимость страховки на здоровье.

В файле [pred_insurance.ipynb](https://github.com/yana-b27/prediction_insurance_cost/blob/main/eda_and_prediction_model/pred_insurance.ipynb) находится информация о разведочном анализе данных и о построении модели линейной регрессии. В файле [model_file.py](https://github.com/yana-b27/prediction_insurance_cost/blob/main/model_file.py) - предварительная обработка данных и предсказательная модель, собранные в виде функций, агрегирующих процесс предсказания. В файле [model_app.py](https://github.com/yana-b27/prediction_insurance_cost/blob/main/model_app.py) содержится скрипт приложения streamlit, который импортирует функции из модуля model_file. В папке [data](https://github.com/yana-b27/prediction_insurance_cost/tree/main/data) находится файл с коэффициентами модели регрессии model_weights.mw и файлы изображений, использованные для создания приложения streamlit. Изображение было получено по [ссылке](https://unsplash.com/photos/KltoLK6Mk-g)
