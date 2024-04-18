import pandas as pd
import numpy as np

class DataPrepKit:
    def __init__(self, file_link):
        self.file_link = file_link
        self.data_reading = DataReading(file_link)
        self.read_file()
        self.summary = Summary()
        self.missing_values_handler = None
        self.categorical_encoder = None

    def read_file(self):
        self.data = self.data_reading.read_file()
        print(self.data)
        return self.data

    def numeric_data_summary(self):
        if self.data is None:
            raise ValueError("No data available. Please read a file first.")

        self.summary.set_data(self.data)
        summary = self.summary.basic_summary()
        print(summary)
        return summary

    def statistics(self):
        if self.data is None:
            raise ValueError("No data available. Please read a file first.")

        self.summary.set_data(self.data)
        statistics = self.summary.calculate_statistics()
        print(statistics)
        return statistics

    def handle_missing_values(self, strategy='mean'):
        if self.data is None:
            self.read_file()
        self.missing_values_handler = MissingValuesHandler(self.data)
        if strategy == 'drop':
            print(self.missing_values_handler.drop())
        elif strategy in ['mean', 'median', 'mode']:
            print(self.missing_values_handler.impute(strategy))
        else:
            raise ValueError("Invalid missing value handling strategy.")

    def encode_categorical_data(self):
        if self.data is None:
            self.read_file()
        self.categorical_encoder = CategoricalEncoder(self.data)
        self.categorical_encoder.encode_categorical_data()
        self.data = self.categorical_encoder.data

class DataReading:
    def __init__(self, file_link):
        self.file_link = file_link
        self.data = None

    def read_file(self):
        file_extension = self.file_link.split('.')[-1].lower()
        if file_extension == 'csv':
            self.data = pd.read_csv(self.file_link)
        elif file_extension == 'json':
            self.data = pd.read_json(self.file_link)
        elif file_extension in ['xls', 'xlsx']:
            self.data = pd.read_excel(self.file_link)
        else:
            raise ValueError("Unsupported file type.")
        return self.data


class Summary:
    def __init__(self):
        self.data = None

    def set_data(self, data):
        self.data = data

    def basic_summary(self):
       if self.data is None:
           raise ValueError("No data available. Please read a file first.")
       return self.data.describe()


    def calculate_statistics(self):
        if self.data is None:
            raise ValueError("No data available. Please read a file first.")

        numeric_columns = self.data.select_dtypes(include=['number'])

        statistics = pd.DataFrame({
            'Average': numeric_columns.mean(),
            'Std Deviation': np.std(numeric_columns, axis=0),
            'Mode': self.data.mode().iloc[0]
        })

        return statistics


class MissingValuesHandler:
    def __init__(self, data):
        self.data = data

    def drop(self):
        cleaned_data = self.data.dropna()
        return cleaned_data

    def impute(self, strategy='mean'):
        numeric_columns = self.data.select_dtypes(include=['number'])
        if numeric_columns.empty:
            raise ValueError("No numeric columns found for imputation.")

        if strategy == 'mean':
            imputed_data = self.data.fillna(numeric_columns.mean())
        elif strategy == 'median':
            imputed_data = self.data.fillna(numeric_columns.median())
        elif strategy == 'mode':
            imputed_data = self.data.fillna(numeric_columns.mode().iloc[0])
        else:
            raise ValueError("Invalid imputation strategy.")

        return imputed_data

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class CategoricalEncoding:
    def __init__(self, data):
        self.data = data

    def one_hot_encode(self, column_name):
        encoder = OneHotEncoder(sparse=False, drop='if_binary')
        encoded_data = encoder.fit_transform(self.data[[column_name]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column_name]))
        self.data = pd.concat([self.data, encoded_df], axis=1)
        self.data.drop([column_name], axis=1, inplace=True)  # Drop the original column

    def label_encode(self, column_name):
        encoder = LabelEncoder()
        self.data[column_name] = encoder.fit_transform(self.data[column_name])

    def get_encoded_data(self):
        return self.data