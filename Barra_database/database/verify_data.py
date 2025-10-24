from pymongo import MongoClient

# --- MongoDB Configuration ---
MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
DB_NAME = "barra_financial_data"
COLLECTION_NAME = "daily_prices"

def get_unique_stock_count():
    """Connects to MongoDB and gets the count of unique ts_codes."""
    try:
        # Connect to the database
        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        print(f"Connected to MongoDB. Checking collection: '{COLLECTION_NAME}'")

        # --- Use the distinct() method ---
        # This is the most direct way to get a list of all unique values for a field.
        unique_ts_codes = collection.distinct("ts_code")
        
        # The number of unique stocks is simply the length of this list.
        count = len(unique_ts_codes)
        
        print(f"\n>>> The number of unique 'ts_code' values is: {count}")
        
        # (Optional) Print the first 10 unique codes
        if count > 0:
            print(f"Sample of unique codes: {unique_ts_codes[:10]}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == '__main__':
    get_unique_stock_count()