from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from peft import LoraConfig, PeftConfig
import bitsandbytes as bnb
import torch
from tqdm import tqdm
from collections import Counter

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

load_dotenv()

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
hf_token = os.getenv("HF_TOKEN")
adapter = "./trained-model"
hub_id = "yishbb"

def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**input_ids, max_new_tokens=1, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.decode(outputs[0])
        answer = result.split("=")[-1].lower()
        # print(answer)
        if "pos" in answer:
            y_pred.append("1")
        elif "negative" in answer:
            y_pred.append("-1")
        elif "neut" in answer:
            y_pred.append("0")
        else:
            y_pred.append("0")
    return y_pred

def load_data(filename: str):
    cols = ["Date", "title", "content", "Open", "High", "Low", "Close", "Volume"]
    df = pd.read_csv(filename, usecols=cols)
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by="date").reset_index(drop=True)
    return df

def merge_with_reliable(df1, filename2: str):
    df2 = pd.read_csv(filename2)
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2.rename(columns={'Date': 'date'}, inplace=True)
    df = df1.merge(df2, on="date", how="inner")
    df = df[["date", "Open", "High", "Low", "Close", "Volume", "sentiment_nltk"]]
    df.columns = [col.lower() for col in df.columns]
    return df

def add_sentiment_nltk(df):
    # Download NLTK resources if not already downloaded
    nltk.download('vader_lexicon')

    # Initialize NLTK's sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Function to get sentiment score
    def get_sentiment_score(text):
        scores = sid.polarity_scores(text)
        # Normalize the compound score to range from 0 to 1
        # Map sentiment scores to 1, -1, or 0
        if scores['compound'] > 0.05:  # Positive sentiment
            return 1
        elif scores['compound'] < -0.05:  # Negative sentiment
            return -1
        else:  # Neutral sentiment
            return 0

    df['sentiment_nltk'] = df['content'].apply(get_sentiment_score)
    return df

def add_sentiment_mistral(df):
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        quantization_config=bnb_config, 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    print(f"[2/5] Loading adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter, device_map="auto")

    print("[3/5] Merge base model and adapter")
    model = model.merge_and_unload()

    df["text"] = pd.DataFrame(df.apply(_generate_test_prompt, axis=1), columns=["title"])
    y_pred = predict(df, model, tokenizer)

    df["sentiment_mistral"] = y_pred
    print(df)

    # print(f"[4/5] Saving model and tokenizer in {hub_id}")
    # model.save_pretrained(f"{args.hub_id}")
    # tokenizer.save_pretrained(f"{args.hub_id}")

    # print(f"[5/5] Uploading to Hugging Face Hub: {hub_id}")
    # model.push_to_hub(hub_id, use_temp_dir=False)
    # tokenizer.push_to_hub(hub_id, use_temp_dir=False)

def _generate_test_prompt(data):
    return f"""
            Your task is to categorize the sentiment of the news headline enclosed in square brackets into one of the following predefined categories:
            
            positive
            neutral
            negative

            You will only respond with the predefined category after = sign without space. Do not include the word "Category". Do not provide explanations or notes.
            [INST][{data}][/INST]=
            """.strip()

def aggregate_sentiment(arr):
    '''
    Averages the sentiment value across the same day, 
    '''

    # return the most frequent
    count = Counter(arr)
    pos, neu, neg = count.get(1, 0), count.get(0, 0), count.get(-1, 0)
    most_count = max(pos, neu, neg)

    # pos and neg count are the same
    if most_count == pos and most_count == neg: 
        return 0  

    elif most_count == pos:
        return 1

    elif most_count == neg:
        return -1
    
    else:
        return 0

def add_up(df):
    '''
    Add a column whether the closing price has gone up compared to the previous day.
    '''
    df["up"] = 0
    df.loc[df["close"].diff() > 0, "up"] = 1
    return df

def collapse_by_date(df):
    '''
    There are multiple entries for each days for news headline.
    '''
    df = df.groupby('date').agg({
        col: "first" if col not in ["sentiment_nltk"] else aggregate_sentiment for col in df.columns
    })
    df = df.reset_index(drop=True) 
    return df
    
if __name__ == "__main__":
    filename1, filename2 = "./data/yahoo_news.csv", "./data/AAPL.csv"
    df = load_data(filename1)
    df = collapse_by_date(df)
    df = add_sentiment_nltk(df)
    # df = add_sentiment_mistral(df)
    df = merge_with_reliable(df, filename2)
    df = add_up(df)
    df.to_csv("./data/yahoo_news_preprocessed.csv", index=False)