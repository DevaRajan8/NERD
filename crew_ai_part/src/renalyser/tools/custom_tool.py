import pandas as pd
import requests

class DatasetValidationAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def validate_dataset(self, dataset: pd.DataFrame) -> str:
        dataset_summary = dataset.describe().to_dict()
        prompt = f"The dataset has the following statistical summary:\n{dataset_summary}\n\nPlease analyze this summary and check for any anomalies, trends, or unusual observations. Provide insights."
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as err:
            return f"Error during dataset validation: {err}"

class DataCleaningAgent:
    def clean_dataset(self, dataset: pd.DataFrame) -> dict:
        cleaning_log = {}
        before_rows = len(dataset)
        dataset = dataset.drop_duplicates()
        after_rows = len(dataset)
        cleaning_log["Removed Duplicates"] = before_rows - after_rows

        num_cols = dataset.select_dtypes(include=["float64", "int64"]).columns
        for column in num_cols:
            missing_count = dataset[column].isnull().sum()
            if missing_count > 0:
                dataset[column] = dataset[column].fillna(dataset[column].mean())
                cleaning_log[f"Filled Missing Values (Numeric): {column}"] = missing_count

        cat_cols = dataset.select_dtypes(include=["object"]).columns
        for column in cat_cols:
            missing_count = dataset[column].isnull().sum()
            if missing_count > 0:
                dataset[column] = dataset[column].fillna("Unknown")
                cleaning_log[f"Filled Missing Values (Categorical): {column}"] = missing_count

        return {"cleaned_dataset": dataset, "cleaning_log": cleaning_log}