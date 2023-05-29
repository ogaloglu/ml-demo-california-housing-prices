# ML Demo for California Housing Prices Dataset
## Overview
A demo for ML based predictions using a XGBoost [[1]](#references) model trained on [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) dataset. [Gradio](https://gradio.app/) is used as the Web UI for presenting the demo and [WandB](https://wandb.ai/site) is used for managing experimentation with XGBoost.
## Project Structure
```bash
ml-demo-california-housing-prices/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── nn_model.py
│   ├── evaluate.py
│   ├── train_xgboost.py
│   └── utils.py
├── models/
│   └── ...
├── data/
│   └── ...
├── notebooks/
│   ├── data_exploration.ipynb
│   └── modeling.ipynb
├── app.py
├── .dockerigj0re
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
docker run -d --name gradio-app-container -p 7860:7860 gradio_app
```
Access the Gradio interface by navigating to [http://127.0.0.1:7860/](http://127.0.0.1:7860/) in your browser. This interface provides a user-friendly way to call and test the ML model directly from your browser.
### Training and Evaluation
Jupyter notebooks are used for training and evaluation. To use them perform the following steps:
```bash
conda create -n "california-housing-task" python=3.10
```
Activate the conda environment with the following command:
```bash
conda activate california-housing-task
```
Install the package by running:
```bash
pip install -e .
```
**(Optional)** Install torch for experimenting with PyTorch.
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then, use following notebooks for model training as well as data exploration:
```bash
notebooks/data_exploration.ipynb
notebooks/modeling.ipynb
```
Alternatively, use the following script to train a XGBoost model:
```bash
python src/train_xgboost.py
```
End evaluate the best model on the test partition by:
```bash
python src/evaluate.py --model_file "best_model.joblib"
```
## Dataset
[California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) Dataset is used for this project.

## Models
* Linear Regression
* Random Forest
* XGBoost
* FFNN
## Metrics
In order to quantify our model’s performance,RMSE and R² metrics are used.

RMSE indicates the absolute fit of the model to the data and hence, how close the observed data points are to the model’s predicted values. Lower values of RMSE indicate a better fit.

R² is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model.
## Results
The final perfornce of best performing 2 models on Test partition is as follows:
* Linear Regression
    * R2 score: 0.6081, RMSE: 71664.08
* Feed Forward Neural Network:
    * R2 score: 0.7541, RMSE: 56768.78
* Random Forest
    * R2 score: 0.8058, RMSE: 50446.14
* XGBoost
    * R2 score: 0.857, RMSE: 43292.66

Results are as expected as tree ensemble methods such as Random Forst or XGBoost outperform other approaces on tabular data, while XGBoost being superior [[2]](#references).
## Reasoning
### Dataset
It is a popular ML dataset because it requires data preprocessing (imputation, handling categorical features), has an easily understandable list of variables and in terms of size, being neither overly simplistic nor excessively complex, it is optimal for practicing ML.
### Models
* Linear regression is a good baseline for tabular data when the outcome variable is continuous.
* As previously stated, tree ensemble methods such as Random Forst or XGBoost are used often for tabular data as they outperform other approaces on such data [[2]](#references).
* It can be interesting to compare the performance of tree ensemble models and neural networks.

Because of more couraging results, the focus was given to XGBoost and training pipeline is created speficially for XGBoost. Because of the inferior results, extending the evaluation pipeline for PyTorch models is omitted.
### Gradio
Gradio is designed with a primary focus on machine learning models. Therefore, if the intention is to develop a web UI specifically for a machine learning model, Gradio offers a straightforward syntax and setup that is well-suited for this purpose.
## Future Work
#### Unit Testing
A testing suite would be helpful for maintaining the repository.
#### Github Actions
GitHub Actions can be utilized for various purposes, such as pushing the Docker image to DockerHub and running tests when changes are committed to the repository.
#### Training Pipeline
Training pipeline can be extended to include models other than XGBoost.
#### Evaluation Pipeline
Current evaluation pipeline can be used only with Sklearn model. The pipeline can be extended such that PyTorch models can be also used.
## References
1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939785
2. Ravid Shwartz-Ziv, & Amitai Armon. (2021). Tabular Data: Deep Learning is Not All You Need.

