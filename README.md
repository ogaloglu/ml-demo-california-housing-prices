# ML Demo for California Housing Prices Dataset
## Overview

## Project Structure
```bash
california-housing-task/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── nn_model.py
│   └── utils.py
├── models/
│   └── ...
├── data/
│   └── ...
├── notebooks/
│   ├── data_exploration.ipynb
│   └── modeling.ipynb
├── app.py
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── setup.py
```
## Usage
To get started, follow the steps below:

Clone the repository by running the following command:
```bash
git clone https://github.com/ogaloglu/ml-demo-california-housing-prices.git
```
Change your current directory to `ml-demo-california-housing-prices` using the command:
```bash
cd ml-demo-california-housing-prices
```
### Running Docker Application
Build the Docker image by executing the following command:
```bash
docker build -t gradio_app:latest .
```
Launch a container based on the built image using the command:
```bash
docker run --name gradio-app-container -p 7860:7860 gradio_app
```
Access the Gradio interface by navigating to [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your browser. This interface provides a user-friendly way to call and test the ML model directly from your browser.
### Training and Evaluation
Jupyter notebooks are used for training and evaluation. To use them perform the following steps:
```bash
conda create -n "california-housing-task" python=3.10
```
Activate the conda environment with the following command:
```bash
conda california-housing-task
```
Install the package by running:
```bash
pip install -e .
```
Then use following notebooks for model training as well as data exploration:
```bash
notebooks/data_exploration.ipynb
notebooks/modeling.ipynb
```
## Dataset
## Models
## Metrics
In order to quantify our model’s performance, we look at the RMSE and Adjusted R² of our model.

RMSE indicates the absolute fit of the model to the data and hence, how close the observed data points are to the model’s predicted values. Lower values of RMSE indicate a better fit. The values for RMSE we get through our pricing model might be comparatively higher as they’ll be in dollars but we’ll be looking out for the least possible option.

Adjusted R² like simple R² describes how ‘good’ the model is at making predictions. However, adding more independent variables or predictors to a regression model tends to increase the R-squared value, which raises the temptation to add even more variables resulting in overfitting. Hence, Adjusted R-squared is preferred to determine how reliable the correlation between the predicted and actual values of the target variable is and how much it is determined by the addition of independent variables in order to avoid overfitting.
## Results
## Reasoining
