#import requests
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import pandas as pd
#import time
from bs4 import BeautifulSoup
from collections import defaultdict
from pprint import pprint
import math
import html
from chromadb import HttpClient
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory
import sqlite3
import spacy
#from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import re
DetectorFactory.seed = 0  # For reproducibility
nlp = spacy.load("en_core_web_sm")

class PreDataFrameProcessor:
    def __init__(self, data_list, title_key, content_key, value_key, language):
        self.data_list = data_list
        self.title_key = title_key
        self.content_key = content_key
        self.value_key = value_key
        self.language = language

    def check_same_type(self, data_list, key):
        if not data_list:
            print("It is an empty list.")
            return
        first_type = type(data_list[0].get(key))
        if all(isinstance(item.get(key), first_type) for item in data_list):
            print(f"All items have the same type for '{key}': {first_type.__name__}")
        else:
            print(f"Items have different types for '{key}'")

    def extract_monolingual_content(self):
        filtered_titles = [
            item[self.title_key][self.language] for item in self.data_list
            if item.get(self.title_key) and self.language in item[self.title_key] and item[self.title_key][self.language].strip() != ""
        ]
        filtered_content = [
            item[self.content_key][self.language] for item in self.data_list
            if item.get(self.content_key) and self.language in item[self.content_key] and item[self.content_key][self.language].strip() != ""
        ]
        self.check_same_type(filtered_titles, self.value_key)
        self.check_same_type(filtered_content, self.value_key)
        return filtered_titles, filtered_content

    def html2str(self, data_list):
        cleaned_contents = []
        for item in data_list:
            html_content = item.get(self.value_key, '')
            if html_content is None:
                html_content = ''
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            cleaned_text = html.unescape(text_content).strip()
            cleaned_contents.append(cleaned_text)
        return cleaned_contents

    def create_dataframe(self, filtered_titles, cleaned_contents):
        cleaned_texts = self.html2str(cleaned_contents)
        data = [{self.title_key: filtered_titles[i][self.value_key], self.content_key: cleaned_texts[i]} for i in range(len(cleaned_contents))]
        return pd.DataFrame(data)

class DataFrameProcessor:
    def __init__(self, df, title_key, content_key, value_key, language):
        self.df = df
        self.title_key = title_key
        self.content_key = content_key
        self.value_key = value_key
        self.language = language

    
    def preprocess_text(self, text, lemmatize_and_remove_stopwords=False):
        preprocessed_text = text.lower()
        # Remove punctuation (but keep numbers and letters)
        # Only remove characters that are not letters, numbers, or whitespace
        preprocessed_text = re.sub(r"[^\w\s]", " ", preprocessed_text)
        # Remove extra whitespace
        preprocessed_text = re.sub(r"\s+", " ", preprocessed_text).strip()
        # Process with spaCy
        doc = nlp(preprocessed_text)
        # Lemmatize, remove stopwords, keep alphabetic and numeric tokens, remove short tokens
        
        if lemmatize_and_remove_stopwords:
            preprocessed_text = [token.lemma_ for token in doc if not token.is_stop]
        
        return preprocessed_text
    
    def is_english(self, text):
        try:
            return detect(text) == 'en'
        except Exception as e:
            print(f"Language detection error: {e}")
            return False
        

    def save_to_file(self, filename, empty_titles, empty_content, both_empty, title_duplicates, content_duplicates, both_duplicates, non_english_titles, non_english_content, both_non_english, total_rows):
        with open(filename, "w") as f:
            f.write(f"Total rows: {total_rows}\n")
            f.write(f"Empty titles: {empty_titles}\n")
            f.write(f"Empty content: {empty_content}\n")
            f.write(f"Title duplicates: {title_duplicates}\n")
            f.write(f"Content duplicates: {content_duplicates}\n")  
            f.write(f"Non-English titles: {non_english_titles}\n")
            f.write(f"Non-English content: {non_english_content}\n")
            f.write(f"Rows with both title and content empty: {both_empty}\n")
            f.write(f"Rows with both title and content duplicated: {both_duplicates}\n")
            f.write(f"Rows with both title and content non-English: {both_non_english}\n")
    
    def save_en_clean_df(self, clean_df, filename="en_clean_df.csv"):
        clean_df.to_csv(filename, index=False)

    def preprocess_df(self, lemmatize_and_remove_stopwords=False, stats_filename=None, english_df_filename=None):
        df[self.title_key] = df[self.title_key].str.strip()
        df[self.content_key] = df[self.content_key].str.strip()
        empty_titles = df[self.title_key].isna().sum() + (df[self.title_key] == '').sum()
        empty_content = df[self.content_key].isna().sum() + (df[self.content_key] == '').sum()
        both_empty = ((df[self.title_key] == '') & (df[self.content_key] == '')).sum()
        print(f"Empty titles: {empty_titles}")
        print(f"Empty content: {empty_content}")
        print(f"Rows with both title and content empty: {both_empty}")

        title_duplicates = df.duplicated(subset=[self.title_key]).sum()
        content_duplicates = df.duplicated(subset=[self.content_key]).sum()
        both_duplicates = df.duplicated(subset=[self.title_key, self.content_key]).sum()
        print(f"Title duplicates: {title_duplicates}")
        print(f"Content duplicates: {content_duplicates}")
        print(f"Rows with both title and content duplicated: {both_duplicates}")
        
        
        df = df[(df[self.title_key] != '') & (df[self.content_key] != '')]
        df.dropna(subset=[self.title_key, self.content_key], inplace=True)
        df.drop_duplicates(subset=[self.title_key, self.content_key], inplace=True)  # Remove rows with NaN in 'title' or 'content'
        
        
        is_title_english = df[self.title_key].apply(self.is_english)
        is_content_english = df[self.content_key].apply(self.is_english)
        non_english_titles = (~is_title_english).sum()
        non_english_content = (~is_content_english).sum()
        both_non_english = ((~is_title_english) & (~is_content_english)).sum()
        print(f"Non-English titles: {non_english_titles}")
        print(f"Non-English content: {non_english_content}")
        print(f"Rows with both title and content non-English: {both_non_english}")
        df["is_english"] = df[self.title_key].apply(self.is_english) & df[self.content_key].apply(self.is_english)
        english_df = df[df["is_english"]].drop(columns=["is_english"]) # Filter English articles

        if stats_filename:
            self.save_to_file(
                stats_filename,
                empty_titles,
                empty_content,
                both_empty,
                title_duplicates,
                content_duplicates,
                both_duplicates,
                non_english_titles,
                non_english_content,
                both_non_english,
                len(english_df),
            )
        
        
        english_df.loc[:, self.title_key] = english_df[self.title_key].apply(self.preprocess_text(lemmatize_and_remove_stopwords=lemmatize_and_remove_stopwords))
        english_df.loc[:, self.content_key] = english_df[self.content_key].apply(self.preprocess_text(lemmatize_and_remove_stopwords=lemmatize_and_remove_stopwords))
        if english_df_filename:
            self.save_english_df(english_df, english_df_filename)
        return english_df


