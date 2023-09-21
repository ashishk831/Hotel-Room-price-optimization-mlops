import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Hotel Room Price estimation Pipeline with ZenML")

    #high_level_image = Image.open("_assets/high_level_overview.png")
    #st.image(high_level_image, caption="High Level Pipeline")

    #whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    #st.markdown(
    """ 
    #### Problem Statement 
     The objective here is to predict the Hotel Room Price for a given order based on features like Hotel type, lead booking time, Deposit type, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the Hotel Room Price.    """
    #)
    #st.image(whole_pipeline_image, caption="Whole Pipeline")
    #st.markdown(
    """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    #)

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the Hotel Room Price. You can input the features listed below and get the Room Price. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Hotel Type | Two types of hotel room available one is resort hotel another city hotel. | 
    | Lead Time   | Number of days that elapsed between the entering date of the booking into the PMS and the arrival date . |  
    | Arrival year | Year of arrival date. | 
    | Arrival month | Month of arrival date with 12 categories: “January” to “December”. |
    | Arrival date | Day of the month of the arrival date.  | 
    | Meal |  Type of meal booked. |
    | Coutry |  Country of origin. |
    | Marget segment |    Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”. |
    | Reserved room type | Code of room type reserved. Code is presented instead of designation for anonymity reasons. | 
    | Assigned room type |  Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons. |
    | Deposit Type |  Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories: No Deposit,Non Refund,Refundable. |
    | Days is waiting list |Number of days the booking was in the waiting list before it was confirmed to the customer. |
    | Customer type | Type of booking, assuming one of four categories: Contract, Group, Transient, Transient-party. |
    | Required car parking | Number of car parking spaces required by the customer. |
    | Special requirement | Number of special requests made by the customer (e.g. twin bed or high floor) |
    | Total stay | Total Number of nights (Sunday to Saturday) the guest stayed or booked to stay at the hotel|
    | Total person | Number of adult including children and babies|
    """
    )
    hotel = st.sidebar.slider("Hotel Type select City Hotel(0) or Resort Hotel(1)",0,1)
    lead_time = st.sidebar.slider("Lead Time")
    arrival_date_year = st.slider("Arrival year",2015,2016,2017)
    arrival_date_month = st.slider("Arrival month",min_value=1,max_value=12,step=1)
    arrival_date_day_of_month = st.slider("Arrival date",min_value=1,max_value=31,step=1)
    meal = st.sidebar.slider("Meal",min_value=1,max_value=5,step=1)
    country = st.slider("Coutry Code",min_value=1,max_value=174,step=1)
    market_segment = st.slider("Marget segment code",min_value=1,max_value=7,step=1)
    reserved_room_type = st.slider("Reserved room type",min_value=1,max_value=9,step=1)
    assigned_room_type = st.slider("Assigned room type",min_value=1,max_value=10,step=1)
    deposit_type = st.slider("Deposit Type",min_value=1,max_value=3,step=1)
    days_in_waiting_list = st.slider("Days is waiting list",)
    customer_type = st.slider("Customer type",min_value=1,max_value=4,step=1)
    required_car_parking_spaces = st.slider("Required car parking",min_value=1,max_value=5,step=1)
    total_of_special_requests = st.sidebar.slider("Special requirement",min_value=1,max_value=6,step=1)
    total_stay = st.sidebar.slider("Total stay")
    total_person = st.sidebar.slider("Total person")

    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "hotel": [hotel],
                "lead_time": [lead_time],
                "arrival_date_year": [arrival_date_year],
                "arrival_date_month": [arrival_date_month],
                "arrival_date_day_of_month": [arrival_date_day_of_month],
                "meal": [meal],
                "country": [country],
                "market_segment": [market_segment],
                "reserved_room_type": [reserved_room_type],
                "assigned_room_type": [assigned_room_type],
                "deposit_type": [deposit_type],
                "days_in_waiting_list": [days_in_waiting_list],
                "customer_type": [customer_type],
                "required_car_parking_spaces": [required_car_parking_spaces],
                "total_of_special_requests": [total_of_special_requests],
                "total_stay": [total_stay],
                "total_person": [total_person]
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)*total_stay*total_person
        st.success(
            "Average Daily Rate  :-{}".format(
                pred
            )
        )
    #if st.button("Results"):
    #    st.write(
    #        "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
    #    )

    #    df = pd.DataFrame(
    #        {
    #            "Models": ["LightGBM", "Xgboost"],
    #            "MSE": [1.804, 1.781],
    #            "RMSE": [1.343, 1.335],
    #        }
    #    )
    #    st.dataframe(df)

    #    st.write(
    #        "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
    #    )
    #    image = Image.open("_assets/feature_importance_gain.png")
    #    st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()