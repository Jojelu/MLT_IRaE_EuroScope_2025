import pandas as pd
from bs4 import BeautifulSoup
import html
from langdetect import detect, DetectorFactory
import spacy
from boilerpy3 import extractors
DetectorFactory.seed = 0  # For reproducibility
nlp = spacy.load("en_core_web_sm")

class PreDataFrameProcessor:
    def __init__(self, data_list, title_key, content_key, value_key, language):
        self.data_list = data_list
        self.title_key = title_key
        self.content_key = content_key
        self.value_key = value_key
        self.language = language

    def check_same_type(self, data_list):
        if not data_list:
            print("It is an empty list.")
            return
        first_type = type(data_list[0])
        if all(isinstance(item, first_type) for item in data_list):
            print(f"All items have the same type: {first_type.__name__}")
        else:
            print("Items have different types.")

    def extract_monolingual_content(self):
        filtered_titles = []
        filtered_content = []
        for item in self.data_list:
            try:
                title = item[self.title_key][self.language][self.value_key]
                content = item[self.content_key][self.language][self.value_key]
                filtered_titles.append(title)
                filtered_content.append(content)
            except (KeyError, TypeError):
                continue  # Skip items missing expected structure
        self.check_same_type(filtered_titles)
        self.check_same_type(filtered_content)
        return filtered_titles, filtered_content

    def html2str(self, data_list):
        cleaned_contents = []
        for html_content in data_list:
            if html_content is None:
                html_content = ''
            # Use BoilerPy3 to extract main content
            try:
                main_content = extractors.get_content(html_content)
            except Exception as e:
                # Fallback: extract raw text if BoilerPy3 fails
                soup = BeautifulSoup(html_content, 'html.parser')
                main_content = soup.get_text()
            #soup = BeautifulSoup(html_content, 'html.parser')       
            #text_content = soup.get_text()
            cleaned_text = html.unescape(main_content).strip()
            cleaned_contents.append(cleaned_text)
        return cleaned_contents

    def create_dataframe(self, filtered_titles, cleaned_contents):
        cleaned_texts = self.html2str(cleaned_contents)
        data = [{self.title_key: filtered_titles[i], self.content_key: cleaned_texts[i]} for i in range(len(cleaned_contents))]
        return pd.DataFrame(data)

class DataFrameProcessor:
    def __init__(self, df, title_key, content_key, value_key, language):
        self.df = df
        self.title_key = title_key
        self.content_key = content_key
        self.value_key = value_key
        self.language = language

    
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

    def preprocess_df(self, stats_filename=None, english_df_filename=None):
        self.df[self.title_key] = self.df[self.title_key].str.strip()
        self.df[self.content_key] = self.df[self.content_key].str.strip()
        empty_titles = self.df[self.title_key].isna().sum() + (self.df[self.title_key] == '').sum()
        empty_content = self.df[self.content_key].isna().sum() + (self.df[self.content_key] == '').sum()
        both_empty = ((self.df[self.title_key] == '') & (self.df[self.content_key] == '')).sum()
        print(f"Empty titles: {empty_titles}")
        print(f"Empty content: {empty_content}")
        print(f"Rows with both title and content empty: {both_empty}")

        title_duplicates = self.df.duplicated(subset=[self.title_key]).sum()
        content_duplicates = self.df.duplicated(subset=[self.content_key]).sum()
        both_duplicates = self.df.duplicated(subset=[self.title_key, self.content_key]).sum()
        print(f"Title duplicates: {title_duplicates}")
        print(f"Content duplicates: {content_duplicates}")
        print(f"Rows with both title and content duplicated: {both_duplicates}")
        
        
        self.df = self.df[(self.df[self.title_key] != '') & (self.df[self.content_key] != '')]
        self.df.dropna(subset=[self.title_key, self.content_key], inplace=True)
        self.df.drop_duplicates(subset=[self.title_key, self.content_key], inplace=True)  # Remove duplicates based on both title and content
        
        is_title_english = self.df[self.title_key].apply(self.is_english)
        is_content_english = self.df[self.content_key].apply(self.is_english)
        non_english_titles = (~is_title_english).sum()
        non_english_content = (~is_content_english).sum()
        both_non_english = ((~is_title_english) & (~is_content_english)).sum()
        print(f"Non-English titles: {non_english_titles}")
        print(f"Non-English content: {non_english_content}")
        print(f"Rows with both title and content non-English: {both_non_english}")
        self.df["is_english"] = self.df[self.title_key].apply(self.is_english) & self.df[self.content_key].apply(self.is_english)
        english_df = self.df[self.df["is_english"]].drop(columns=["is_english"]) # Filter English articles

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
        
        
        if english_df_filename:
            self.save_en_clean_df(english_df, english_df_filename)
        return english_df


