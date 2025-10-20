import time
import asyncio
import inspect
import pandas as pd
from tqdm import tqdm
from googletrans import Translator

# Settings
FOLDER = "excel_data"
FILE = "Mapping.csv"
TARGET_COL = "Part Name"
LANG = "en"
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0


def translate(translator, text, max_retries, initial_backoff, lang):
    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):

        # Translate text
        try:
            result = translator.translate(text, dest=lang)
            if inspect.isawaitable(result):
                result = asyncio.get_event_loop().run_until_complete(result)
            return result.text

        # Failed to translate text
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR]: Failed to translate text")
                return text
            else:
                print(f"[WARNING]: Trying again to translate text")
                time.sleep(backoff)
                backoff *= 2  # Double backoff timeout


def main():

    # Load source file
    src = FOLDER + "/" + FILE
    df = pd.read_csv(src, encoding="utf-8-sig")
    if TARGET_COL not in df.columns:
        print(f"[ERROR]: Column '{TARGET_COL}' not found in {src}")
        return

    # Translate the untranslated column
    translator = Translator()
    translated_texts = []
    try:
        for text in tqdm(df[TARGET_COL].fillna(""), desc="Translating"):
            translated_texts.append(
                translate(translator, str(text), MAX_RETRIES, INITIAL_BACKOFF, LANG)
            )
    except KeyboardInterrupt:
        print("[WARNING]: Interrupted! Saving in progress…")

    # Save translated column
    df["Translated"] = translated_texts + [""] * (len(df) - len(translated_texts))
    df.to_csv(src, index=False, encoding="utf-8-sig")
    print("Saved results")


if __name__ == "__main__":
    main()
