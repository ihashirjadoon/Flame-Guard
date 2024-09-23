# FlameGuard - Forest Fire Prediction

FlameGuard is a predictive machine learning model designed to forecast forest fires based on various environmental factors. By analyzing data like temperature, humidity, wind speed, and moisture indices, FlameGuard aims to mitigate the risks and impacts of forest fires, providing early warnings to enable faster and more efficient interventions. This project leverages advanced machine learning algorithms to help safeguard ecosystems, wildlife, and human communities from the devastating effects of forest fires.

## Techwiz 5 Global Competition

FlameGuard is being developed as part of the global competition **Techwiz 5**, where teams from around the world compete to create innovative technological solutions to pressing real-world problems. Our team, **MSG HASH**, is proudly participating in this competition from the platform of **Aptech MSG**. The project aims to showcase our expertise in machine learning and predictive analytics by addressing the critical issue of forest fire prediction. Through FlameGuard, we hope to contribute to environmental conservation by leveraging cutting-edge technology to predict and mitigate the risk of forest fires.




## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Functional Requirements](#functional-requirements)
- [Non-Functional Requirements](#non-functional-requirements)
- [Constraints](#constraints)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Forest fires are a growing environmental concern, worsened by climate change and human activities. FlameGuard addresses this issue by predicting the likelihood of forest fires, helping prevent potential disasters. By combining weather patterns, historical fire occurrences, and vegetation data, FlameGuard enables forest stations to respond quickly and mitigate the impact of fires.

## Dataset

The dataset used for this project is available at [Kaggle - Forest Fires Data](https://www.kaggle.com/datasets/elikplim/forest-fires-data-set). It consists of 515 instances with variables such as:

- **Month**: Month of the year
- **Fine Fuel Moisture Code (FFMC)**: Moisture content of surface litter
- **Duff Moisture Code (DMC)**: Moisture content of loosely compacted organic layers
- **Drought Code (DC)**: Moisture content of deep organic layers
- **Initial Spread Index (ISI)**: Spread rate of a fire
- **Temperature**: In degrees Celsius
- **Relative Humidity (RH)**: In percentage
- **Wind Speed**: In km/h
- **Class**: The presence (1) or absence (0) of fire

The data focuses on the northeast region of Portugal and is divided into training, generalization, and testing subsets.

## Key Features

- Predicts forest fire likelihood using environmental factors like temperature, humidity, and wind speed.
- Provides real-time predictions with user inputs via a simple, user-friendly interface.
- Alerts nearby forest stations for timely response and intervention.
- Scalable to handle increasing data volume and user requests.

## Machine Learning Pipeline

1. **Data Collection**: Gather data from the dataset provided.
2. **Data Pre-Processing**: Clean the data to handle missing values and normalize it for modeling.
3. **Exploratory Data Analysis (EDA)**: Visualize and understand data patterns and relationships.
4. **Feature Engineering**: Create new features to enhance model performance.
5. **Feature Selection**: Choose significant features impacting fire predictions.
6. **Model Building**: Train multiple models (Logistic Regression, Decision Trees, Random Forests, Support Vector Machines).
7. **Model Evaluation**: Use metrics like accuracy, precision, recall, and F1-score to assess models.
8. **Hyperparameter Tuning**: Optimize the best-performing model.
9. **Model Serialization**: Save the trained model using Pickle for future predictions.

## Functional Requirements

1. **Data Collection**: Dataset is collected and prepared for analysis.
2. **Data Preprocessing**: Handle missing data, outliers, and normalization.
3. **EDA**: Perform visualization and statistical analysis to uncover patterns.
4. **Feature Engineering**: Create features that improve model performance.
5. **Model Building**: Train models using various machine learning algorithms.
6. **Model Evaluation**: Evaluate models and select the most accurate one.
7. **Hyperparameter Tuning**: Fine-tune the model to improve performance.
8. **Model Serialization**: Save the trained model for future use.
9. **Prediction**: Predict fire risk based on user inputs.

## Non-Functional Requirements

- **Accuracy**: Achieve over 90% prediction accuracy.
- **Security**: Ensure data security and protect against unauthorized access.
- **Robustness**: Handle various input conditions and provide reliable predictions.
- **Scalability**: Capable of handling increased data and user requests.
- **User-Friendliness**: Provide a simple interface for easy data input and prediction.

## Constraints

- **Computational Limitations**: Training large models can be computationally expensive.
- **Regional Variability**: Different environmental conditions can affect prediction accuracy across diverse geographic areas.

## Installation

To run the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/ihashirjadoon/Flame-Guard
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    python app.py
    ```

## Usage

1. Input environmental values such as **Temperature**, **Humidity**, and **Wind Speed** into the user interface.
2. The system will predict the likelihood of a forest fire.
3. If a fire risk is detected, a notification will be sent to the nearby forest station.

## Technologies Used

- **Python** (NumPy, Pandas, Scikit-learn)
- **Machine Learning**: Logistic Regression, Decision Trees, Random Forests, Support Vector Machines
- **Flask** for the web application
- **Pickle** for model serialization
- **Matplotlib** for data visualization

## Contributing

Contributions are welcome! Feel free to submit a pull request or report an issue. Please ensure your contributions align with the project’s goals and coding standards.


