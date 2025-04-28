# Forecasting Life Expectancy (AHH) in Sidoarjo

This project aims to predict **Life Expectancy (AHH)** in Sidoarjo based on birth data of males and females from year to year. It utilizes **Time Series Forecasting** techniques with an **LSTM (Long Short-Term Memory)** model to predict future AHH values.

## Project Description

1. **Data Scraping:**
   - The project begins by scraping birth data based on gender (male and female) obtained from local or public data sources.
   - The collected data is then processed and formatted to be used for further analysis.

2. **Preprocessing:**
   - Initially unstructured or messy data is processed using preprocessing techniques to clean and organize the data.
   - Preprocessing steps include:
     - Normalization of the data using **MinMaxScaler** to bring the values into the same range.
     - Creation of sequence data for LSTM model training.

3. **Modeling:**
   - An **LSTM** (Long Short-Term Memory) model is used to predict Life Expectancy based on historical data.
   - Two LSTM models are built to predict AHH based on gender (male and female).
   - The model is trained with the processed data, and the results are used to make predictions for upcoming years.

4. **Deployment:**
   - The trained model is deployed using **Streamlit** to create an interactive user interface (UI).
   - Users can input a target year for prediction and view the results in the form of graphs and tables.


## Technologies Used

- **Python**: The main programming language for this project.
- **Streamlit**: For deploying the interactive web application.
- **TensorFlow/Keras**: For creating and training the LSTM model.
- **Plotly**: For interactive visualization of the prediction graphs.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For mathematical operations and array manipulation.
- **Scikit-learn**: For preprocessing and model evaluation.
