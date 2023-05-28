# California Housing Prices Dataset
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
git clone 
```
Change your current directory to `california-housing-task` using the command:
```bash
cd california-housing-task
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
## Results
## Reasoining
