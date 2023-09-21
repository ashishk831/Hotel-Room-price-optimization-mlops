import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self,data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocessing data
        """
        try:
            data["agent"].fillna(data["agent"].median(),inplace=True)
            data["children"].replace(np.nan,0, inplace=True)
            data = data.drop(data[data['adr'] < 50].index)
            data = data.drop(data[data['adr'] > 5000].index)
            data["total_stay"] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']            
            data["total_person"] = data["adults"] + data["children"] + data["babies"]  
            data = data.drop(
                [
                    "company",
                    "is_canceled",
                    "arrival_date_week_number",
                    "distribution_channel",
                    "is_repeated_guest",
                    "previous_cancellations",
                    "previous_bookings_not_canceled",
                    "booking_changes",
                    "agent",
                    "reservation_status",
                    "reservation_status_date",
                    "stays_in_week_nights",
                    "stays_in_weekend_nights",
                    "adults","children","babies"
                ],axis=1
            )
            data=data.dropna()
            le = LabelEncoder()
            data['hotel'] = le.fit_transform(data['hotel'])
            data['arrival_date_month'] = le.fit_transform(data['arrival_date_month'])
            data['meal'] = le.fit_transform(data['meal'])
            data['country'] = le.fit_transform(data['country'])
            data['market_segment'] = le.fit_transform(data['market_segment'])
            data['reserved_room_type'] = le.fit_transform(data['reserved_room_type'])
            data['assigned_room_type'] = le.fit_transform(data['assigned_room_type'])
            data['deposit_type'] = le.fit_transform(data['deposit_type'])
            data['customer_type'] = le.fit_transform(data['customer_type'])

            return data
    
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(["adr"],axis=1)
            y = data["adr"]
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)          
            
            return X_train, X_test, y_train, y_test 
        
        except Exception as e:
            logging.error("Error in dividing data:{}".format(e))
            raise e

class DataCleaning:
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    def __init__(self,data: pd.DataFrame,Strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = Strategy
        
    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Handled data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e