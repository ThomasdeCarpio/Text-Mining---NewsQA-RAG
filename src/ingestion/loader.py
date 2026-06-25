import os
import json
import pandas as pd
from typing import List, Dict

class DataLoader:
    """
    A utility class to load raw data from various sources (Local HTML, JSON, CSV, etc.)
    and return it in a standardized format for the cleaner or chunker to process.
    """

    @staticmethod
    def load_local_html_directory(directory_path: str) -> List[Dict[str, str]]:
        """
        Loads all raw HTML files from a local directory.
        
        Args:
            directory_path (str): The path to the folder containing .html files.
            
        Returns:
            List[Dict]: A list of dictionaries, where each dict contains:
                        - 'filename': The name of the file (e.g., 'article_1.html')
                        - 'html_content': The raw HTML string.
        """

        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"The directory {directory_path} does not exist.")

        loaded_files = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".html"):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        loaded_files.append({
                            "filename": filename,
                            "html_content": content
                        })
                except Exception as e:
                    print(f"❌ Error loading file {filename}: {e}")
                    
        return loaded_files

    @staticmethod
    def load_processed_json_directory(directory_path: str) -> List[Dict]:
        """
        Loads all cleaned JSON files (produced by the cleaner) so they can be passed to the chunker.
        
        Args:
            directory_path (str): The path to the folder containing _clean.json files.
            
        Returns:
            List[Dict]: A list of the parsed JSON objects containing 'text' and 'metadata'.
        """
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"The directory {directory_path} does not exist.")

        loaded_data = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Inject the filename as the article_id if it's not present
                        if "article_id" not in data["metadata"]:
                            data["metadata"]["article_id"] = filename.replace("_clean.json", "")
                            
                        loaded_data.append(data)
                except Exception as e:
                    print(f"❌ Error loading JSON {filename}: {e}")
                    
        return loaded_data

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Loads a CSV dataset (useful for loading the evaluation datasets or Multi-News later).
        
        Args:
            file_path (str): The path to the .csv file.
            
        Returns:
            pd.DataFrame: The loaded dataset.
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
        return pd.read_csv(file_path)