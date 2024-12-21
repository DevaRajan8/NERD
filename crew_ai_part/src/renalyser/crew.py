import os
from crewai import Crew, Task, Agent
from tools.custom_tool import DatasetValidationAgent, DataCleaningAgent
from typing import Callable, Any

api_key = os.getenv("GROQ_API_KEY")

class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description

def validate_dataset_func(dataset: Any) -> Any:
    validation_agent = DatasetValidationAgent(api_key)
    return validation_agent.validate_dataset(dataset)

validate_dataset_tool = Tool(
    name="Validate Dataset Tool",
    func=validate_dataset_func,
    description="Validates a dataset using an LLM to identify anomalies, trends, and insights."
)

def clean_dataset_func(dataset: Any) -> Any:
    cleaning_agent = DataCleaningAgent()
    return cleaning_agent.clean_dataset(dataset)

clean_dataset_tool = Tool(
    name="Clean Dataset Tool",
    func=clean_dataset_func,
    description="Cleans a dataset by removing duplicates and handling missing values."
)

dataset_validation_agent = Agent(
    name="DatasetValidationAgent",
    role="Data Validator",
    goal="Analyze the dataset using LLM to identify anomalies or trends and Check if the dataset has any duplicate rows or columns or values.",
    tools=[validate_dataset_tool],
    backstory="You have to use advanced LLM capabilities to validate datasets by analyzing statistical summaries."
)

data_cleaning_agent = Agent(
    name="DataCleaningAgent",
    role="Dataset Cleaner",
    goal="Clean the dataset by removing duplicates and handling missing values and null values and not a number entries.",
    tools=[clean_dataset_tool],
    backstory="You have to focus on ensuring data quality by cleaning datasets through duplicate removal and imputing missing values."
)

validate_task = Task(
    name="Validate Dataset",
    description="Validates a dataset to identify anomalies, trends, and insights using an LLM.",
    agent=dataset_validation_agent,
    expected_output="Anomaly detection and insights from the dataset."
)

clean_task = Task(
    name="Clean Dataset",
    description="Cleans a dataset by removing duplicates and handling missing values.",
    agent=data_cleaning_agent,
    expected_output="Cleaned dataset and cleaning log."
)

class ExtendedCrew(Crew):
    def run_task(self, task_name: str, **kwargs) -> Any:
        task = next((task for task in self.tasks if task.name == task_name), None)
        if not task:
            raise ValueError(f"Task '{task_name}' not found.")
        agent = task.agent
        tool = agent.tools[0]
        if tool and callable(tool.func):
            return tool.func(**kwargs)
        else:
            raise ValueError(f"Tool for task '{task_name}' is not callable.")

crew = ExtendedCrew(
    agents=[dataset_validation_agent, data_cleaning_agent],
    tasks=[validate_task, clean_task]
)