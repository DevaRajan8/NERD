from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Update URI if needed

# Create or access a database
db = client["my_database"]

# Create or access a collection (similar to a table in relational DBs)
collection = db["datasets"]

print("Connected to MongoDB successfully!")
