"""Main app for predicting housing prices."""
import gradio as gr
import pandas as pd

from src.utils import load_file

preprocessor = load_file("preprocessor.joblib")
model = load_file("best_model.joblib")


def predict(
    longitude: float,
    latitude: int,
    housing_median_age: float,
    total_rooms: int,
    total_bedrooms: int,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str,
) -> float:
    """Prediction function for inference."""
    df = pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        },
        index=[0],
    )
    preprocessed_df = preprocessor.transform(df)
    prediction = model.predict(preprocessed_df)

    return round(prediction[0], 2)


with gr.Blocks() as app:
    with gr.Row():
        longitude = gr.Slider(
            -124,
            -115,
            label="Longitude",
            info="A measure of how far west a house is; a higher value is "
            "farther west",
            value=-120,
        )
        latitude = gr.Slider(
            32,
            42,
            label="Latitude",
            info="A measure of how far north a house is; a higher value is "
            "farther north",
            value=37,
        )
    with gr.Row():
        housing_median_age = gr.Slider(
            1,
            52,
            label="Housing Median Age",
            info="Median age of a house within a block; a lower number is a "
            "newer building",
            value=20,
        )
        total_rooms = gr.Slider(
            2,
            40000,
            step=1,
            label="Total Rooms",
            info="Total number of rooms within a block",
            value=20000,
        )
    with gr.Row():
        total_bedrooms = gr.Slider(
            1,
            7000,
            step=1,
            label="Total Bedrooms",
            info="Total number of bedrooms within a block",
            value=3000,
        )
        population = gr.Slider(
            3,
            40000,
            label="Population",
            info="Total number of people residing within a block",
            value=20000.0,
        )
    with gr.Row():
        households = gr.Slider(
            1,
            6082,
            label="Households",
            info="Total number of households, a group of people residing "
            "within a home unit, for a block",
            value=3000.0,
        )
        median_income = gr.Slider(
            0,
            15,
            label="Median Income",
            info="Median income for households within a block of"
            "houses (measured in tens of thousands of US Dollars)",
            value=7.0,
        )
    ocean_proximity = gr.Radio(
        ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"],
        label="Ocean Proximity",
        info="Location of the house w.r.t ocean/sea",
        value="NEAR BAY",
    )

    predict_btn = gr.Button("Predict")
    output = gr.Textbox(
        label="Prediction",
        info="Median house value for households within a block (measured in US"
        " Dollars)",
    )
    predict_btn.click(
        fn=predict,
        inputs=[
            longitude,
            latitude,
            housing_median_age,
            total_rooms,
            total_bedrooms,
            population,
            households,
            median_income,
            ocean_proximity,
        ],
        outputs=output,
        api_name="predict",
    )

app.launch(server_name="0.0.0.0", server_port=7860)
