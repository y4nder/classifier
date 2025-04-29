from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from classifier import classify_difficulty, extract_final_topic
import pandas as pd 
import os 
import topics

tqdm.pandas()
load_dotenv(override=True)
API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=API_KEY)

csv_path = "files/cleaned_questions.csv"
df = pd.read_csv(csv_path)


# Enhanced topic classification function with error handling
def safe_classify_topic(question):
    sample_topics = topics.get_sample_topics()
    try:
        return extract_final_topic(question, topics=sample_topics)
            
    except Exception as e:
        print(f"Error processing question: '{question[:30]}...' - {str(e)}")
        return f"ERROR: {str(e)}"  # Or return None if preferred

# Enhanced difficulty classification function with error handling    
def safe_classify_diff(question):
    try:
        return classify_difficulty(question)
    except Exception as e:
        print(f"Error processing question: '{question[:30]}...' - {str(e)}")
        return f"ERROR: {str(e)}"  

# Apply with progress bar and error handling
print("Starting classification process...")
df['classification'] = df['question'].progress_apply(safe_classify_topic)
print(f"\nClassification completed")

print("Starting difficulty classification process...")
df['difficulty'] = df['question'].progress_apply(safe_classify_diff)
print(f"\nDifficulty Classification completed")


# saving as csv file
output_folder = "files"
output_filename = 'classified_questions.csv'

# Create the folder if it doesn't exist
full_path = os.path.join(output_folder, output_filename)
os.makedirs(output_folder, exist_ok=True)

df.to_csv(full_path, index=False)
# Full path
print(f"\nSaved results to {output_filename}")

