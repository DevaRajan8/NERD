import os
import hashlib
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from crew import crew

api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
if not api_key or not mongo_uri:
    st.warning("API Keys or MongoDB URI not set. Please configure them in the environment.")
    st.stop()

client = MongoClient(mongo_uri)
db = client.research_paper_db
collection = db.datasets

def generate_dataset_hash(dataset: pd.DataFrame) -> str:
    dataset_string = dataset.to_csv(index=False)
    return hashlib.md5(dataset_string.encode()).hexdigest()

st.title("Research Paper to Dataset Validation and Cleaning")
st.sidebar.header("Database Information")
try:
    total_datasets = collection.count_documents({})
    st.sidebar.write(f"**Database Name:** {db.name}")
    st.sidebar.write(f"**Collections:** {', '.join(db.list_collection_names())}")
    st.sidebar.write(f"**Total Datasets:** {total_datasets}")
except Exception as e:
    st.sidebar.error(f"Unable to fetch database stats: {e}")

task_option = st.selectbox("Select Task:", ["Validate Dataset with LLM", "Clean Dataset", "Load Saved Dataset"])

if task_option == "Validate Dataset with LLM":
    uploaded_file = st.file_uploader("Upload a CSV File:", type=["csv"])
    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        dataset_hash = generate_dataset_hash(dataset)
        existing_dataset = collection.find_one({"hash": dataset_hash})
        if existing_dataset:
            st.warning("This dataset has already been added to the database. Skipping database insertion.")
        else:
            st.write("### Uploaded Dataset")
            st.dataframe(dataset)
        if st.button("Validate Dataset"):
            result = crew.run_task("Validate Dataset", dataset=dataset)
            st.write("### Validation Result")
            st.markdown(result)
        if not existing_dataset:
            collection.insert_one({"hash": dataset_hash, "dataset": dataset.to_dict(orient="records")})
            st.success("Dataset validated and saved to MongoDB.")

elif task_option == "Clean Dataset":
    uploaded_file = st.file_uploader("Upload a CSV File:", type=["csv"])
    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        dataset_hash = generate_dataset_hash(dataset)
        existing_dataset = collection.find_one({"hash": dataset_hash})
        if existing_dataset:
            st.warning("This dataset has already been added to the database. Skipping database insertion.")
        else:
            st.write("### Uploaded Dataset")
            st.dataframe(dataset)
        if st.button("Clean Dataset"):
            result = crew.run_task("Clean Dataset", dataset=dataset)
            cleaned_dataset, cleaning_log = result["cleaned_dataset"], result["cleaning_log"]
            st.write("### Cleaned Dataset")
            st.dataframe(cleaned_dataset)
            st.write("### Cleaning Details")
            st.json(cleaning_log)
        if not existing_dataset:
            collection.insert_one({"hash": dataset_hash, "dataset": cleaned_dataset.to_dict(orient="records")})
            st.success("Cleaned dataset saved to MongoDB.")

elif task_option == "Load Saved Dataset":
    datasets = list(collection.find())
    if datasets:
        if "dataset_view" not in st.session_state:
            st.session_state.dataset_view = None
        for i, entry in enumerate(datasets):
            dataset = pd.DataFrame(entry["dataset"])
            st.write(f"### Dataset {i + 1}")
            st.write("**Dataset Preview (First 2 Rows):**")
            st.dataframe(dataset.head(2))
            if st.button(f"View Full Dataset {i + 1}", key=f"view_{i}"):
                st.session_state.dataset_view = i
        if st.session_state.dataset_view is not None:
            selected_dataset = pd.DataFrame(datasets[st.session_state.dataset_view]["dataset"])
            st.write("### Full Dataset")
            st.dataframe(selected_dataset)
    else:
        st.info("No datasets found in the database.")
