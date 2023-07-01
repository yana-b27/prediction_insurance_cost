# prediction_insurance_cost

Данный проект посвящен предсказанию стоимости страховки здоровья с помощью модели линейной регрессии и создания приложения для предсказания с помощью библиотеки streamlit. В проекте использовался датасет из книги "Machine Learning with R", скачанный по [ссылке](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv). В наборе данных содержится следующая информация:

- возраст (поле age)
- пол (поле sex)
- индекс массы тела (поле bmi)
- кол-во детей у респондента (поле children)
- является ли респондент курильщиком или нет (поле smoker)
- регион проживания (поле region)
- стоимость страховки здоровья (поле charges)

Разведочный анализ данных произведен с помощью библиотек pandas и seaborn для построения графиков. Доя подготовки данных к построению модели использовался модуль sklearn для кодирования категориальных признаков, сформирования обучающей и тестовой выборки, в дальнейшем с этим модулем было произведено построение модели линейной регрессии. Для создания приложения применялась библиотека streamlit, pillow для загрузки изображений, plotly для создания интерактивных графиков, описывающих данные
