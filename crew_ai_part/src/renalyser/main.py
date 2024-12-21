import os
import hashlib
import streamlit as st
import pandas as pd
import json
from pymongo import MongoClient
from psycopg2 import connect, sql
from PyPDF2 import PdfReader
from weaviate import Client as WeaviateClient
import spacy
from crew import crew
api_key = os.getenv("GROQ_API_KEY")
weaviate_url = os.getenv("WEAVIATE_URL")
mongo_uri = os.getenv("MONGO_URI")
pg_conn_string = os.getenv("psql")
if not api_key or not mongo_uri or not pg_conn_string or not weaviate_url:
    st.warning("API Keys or MongoDB URI not set. Please configure them in the environment.")
    st.stop()
try:
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client.research_paper_db
    collection = db.datasets
    pg_conn = connect(pg_conn_string)
    pg_conn.autocommit = True
    weaviate_client = WeaviateClient(weaviate_url)
    pg_cursor = pg_conn.cursor()
    nlp = spacy.load("en_core_web_sm")
    def generate_dataset_hash(dataset: pd.DataFrame) -> str:
        dataset_string = dataset.to_csv(index=False)
        return hashlib.md5(dataset_string.encode()).hexdigest()
    def clean_column_names(df):
        import re
        df.columns = [re.sub(r"[^\w]", "_", col)[:230] for col in df.columns]
        return df
    def extract_text_from_pdf(file):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    def extract_named_entities(text):
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        return entities
    def upload_to_weaviate(dataset: pd.DataFrame):
        try:
            class_name = "Dataset"
            if not weaviate_client.schema.contains({"class": class_name}):
                weaviate_client.schema.create_class({
                    "class": class_name,
                    "properties": [{"name": col, "dataType": ["string"]} for col in dataset.columns]
                })

            for _, row in dataset.iterrows():
                data_object = row.to_dict()
                weaviate_client.data_object.create(data_object, class_name)

            st.success("Dataset uploaded to Weaviate successfully!")
        except Exception as e:
            st.error(f"Error uploading to Weaviate: {e}")
    def display_weaviate_contents():
        try:
            class_name = "Dataset"
            if not weaviate_client.schema.contains({"class": class_name}):
                st.warning("No datasets found in Weaviate.")
                return

            results = weaviate_client.data_object.get()
            if results and results.get("objects"):
                st.write("### Contents in Weaviate")
                for obj in results["objects"]:
                    if obj["class"] == class_name:
                        st.json(obj)
            else:
                st.warning("No objects found in Weaviate.")
        except Exception as e:
            st.error(f"Error fetching data from Weaviate: {e}")
    def chunk_text_with_entities_pgai(text, entities):
        try:
            pg_cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT,
                    entities JSONB
                )
            """)
        except Exception as e:
            st.error(f"Error creating table: {e}")
            return "Failed to create table in PostgreSQL."
        chunk_size = 500
        try:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                chunk_entities = [ent for ent in entities if ent['start'] >= i and ent['end'] <= i + chunk_size]
                pg_cursor.execute(
                    sql.SQL("SELECT COUNT(*) FROM text_chunks WHERE chunk_text = %s"),
                    [chunk]
                )
                exists = pg_cursor.fetchone()[0]
                if exists == 0:
                    pg_cursor.execute(
                        sql.SQL("INSERT INTO text_chunks (chunk_text, entities) VALUES (%s, %s)"),
                        [chunk, json.dumps(chunk_entities)]
                    )
        except Exception as e:
            st.error(f"Error inserting chunks into PostgreSQL: {e}")
            return "Failed to insert chunks into PostgreSQL."

        return "Text chunking completed and stored in PostgreSQL."
    def display_chunks_from_postgresql():
        try:
            pg_cursor.execute("SELECT chunk_text, entities FROM text_chunks ORDER BY id;")
            rows = pg_cursor.fetchall()
            if rows:
                st.write("### Retrieved Chunks")
                for idx, (chunk_text, entities) in enumerate(rows, start=1):
                    st.write(f"**Chunk {idx}:**")
                    st.text(chunk_text)
                    st.json(entities)
            else:
                st.warning("No chunks found in the database.")
        except Exception as e:
            st.error(f"Error retrieving chunks from PostgreSQL: {e}")

    # Streamlit App
    st.title("NERD")
    st.sidebar.header("Database Information")
    try:
        total_datasets = collection.count_documents({})
        st.sidebar.write(f"**Database Name:** {db.name}")
        st.sidebar.write(f"**Collections:** {', '.join(db.list_collection_names())}")
        st.sidebar.write(f"**Total Datasets:** {total_datasets}")
    except Exception as e:
        st.sidebar.error(f"Unable to fetch database stats: {e}")

    task_option = st.selectbox("Select Task:", [
        "Validate Dataset with LLM",
        "Clean Dataset",
        "Load Saved Dataset",
        "Process PDF for Chunking",
        "View Chunks in Database",
        "View Weaviate Contents"
    ])
    if task_option == "Validate Dataset with LLM":
        uploaded_file = st.file_uploader("Upload a CSV File:", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)
            dataset = clean_column_names(dataset)
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
                upload_result = upload_to_weaviate(dataset)
                st.success(upload_result)
                st.success("Dataset validated and saved to MongoDB.")
    elif task_option == "Clean Dataset":
        uploaded_file = st.file_uploader("Upload a CSV File:", type=["csv"])
        if uploaded_file:
            dataset = pd.read_csv(uploaded_file)
            dataset = clean_column_names(dataset)
            dataset_hash = generate_dataset_hash(dataset)
            existing_dataset = collection.find_one({"hash": dataset_hash})
            if existing_dataset:
                st.warning("This dataset has already been added to the database. Skipping database insertion.")
            else:
                st.write("### Uploaded Dataset")
                st.dataframe(dataset)
            if st.button("Clean Dataset"):
                result = crew.run_task("Clean Dataset", dataset=dataset)
                if result and "cleaned_dataset" in result and "cleaning_log" in result:
                    cleaned_dataset, cleaning_log = result["cleaned_dataset"], result["cleaning_log"]
                    st.write("### Cleaned Dataset")
                    st.dataframe(cleaned_dataset)
                    st.write("### Cleaning Details")
                    st.json(cleaning_log)
                    if not existing_dataset:
                        collection.insert_one({"hash": dataset_hash, "dataset": cleaned_dataset.to_dict(orient="records")})
                        st.success("Cleaned dataset saved to MongoDB.")
                else:
                    st.error("Error cleaning dataset. Please check the crew task result.")
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
    elif task_option == "Process PDF for Chunking":
        uploaded_file = st.file_uploader("Upload a PDF File:", type=["pdf"])
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            st.write("### Extracted Text")
            st.text(text[:500])

            entities = extract_named_entities(text)
            st.write("### Named Entities")
            st.json(entities)

            result = chunk_text_with_entities_pgai(text, entities)
            display_chunks_from_postgresql()
            st.success(result)
    elif task_option == "View Chunks in Database":
        if st.button("Retrieve Chunks"):
            display_chunks_from_postgresql()
    elif task_option == "View Weaviate Contents":
        if st.button("View Weaviate Contents"):
            display_weaviate_contents()
finally:
    if 'mongo_client' in locals() and mongo_client:
        mongo_client.close()
    if 'pg_cursor' in locals() and pg_cursor:
        pg_cursor.close()
    if 'pg_conn' in locals() and pg_conn:
        pg_conn.close()