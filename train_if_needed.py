import pandas as pd
import os
from model_training import train_model  # Imports your existing training function

def check_and_train():
    DATA_PATH = "aqi_data.csv"
    CHECKPOINT_FILE = ".last_train_count.txt"
    THRESHOLD = 10  # Number of new rows needed to trigger training

    # 1. Check if data exists
    if not os.path.exists(DATA_PATH):
        print("aqi_data.csv not found. Skipping.")
        return

    # 2. Get current row count
    df = pd.read_csv(DATA_PATH)
    current_count = len(df)

    # 3. Get last trained count
    last_count = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            try:
                last_count = int(f.read().strip())
            except ValueError:
                last_count = 0

    # 4. Decision Logic
    new_rows = current_count - last_count
    
    if new_rows >= THRESHOLD:
        print(f"🚀 {new_rows} new rows detected! Starting retraining...")
        
        # Run your actual training function
        train_model()
        
        # Update the checkpoint file
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(str(current_count))
            
        print(f"✅ Checkpoint updated to {current_count} rows.")
    else:
        print(f"☕ Only {new_rows} new rows found. Waiting for {THRESHOLD} rows to retrain.")

if __name__ == "__main__":
    check_and_train()
