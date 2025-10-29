import pandas as pd
import re
from tqdm import tqdm
from preprocess import save_df


# Settings
FOLDER_PATH = "indirect_cc_predictor"
FILE_NAME = "parts_data.csv"
TARGET_COL = "Category L4"
OUTPUT_COL = "Category L4 - Cleaned"
ENFORCED_TEXT_COLS = []


def clean_text(text):
    temp = text.replace("<", " less ").replace(">", " more ")
    cleaned = re.sub(r"[^A-Za-z0-9]", " ", temp)  # keep letters and numbers
    return re.sub(r"\s+", " ", cleaned).strip().lower()  # collapse spaces and lowercase


def clean_column(file_path, column_name):

    # Load source file
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    if column_name not in df.columns:
        print(f"[ERROR] Column '{column_name}' not found in {file_path}")
        return

    # Clean column and save results
    df[OUTPUT_COL] = [
        clean_text(str(txt)) for txt in tqdm(df[column_name], desc="Cleaning")
    ]
    save_df(df, file_path, ENFORCED_TEXT_COLS)


if __name__ == "__main__":
    file_path = FOLDER_PATH + "/" + FILE_NAME
    clean_column(file_path, TARGET_COL)
