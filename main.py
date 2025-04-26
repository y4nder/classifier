from typing import List
import pandas as pd 
import os 

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
tqdm.pandas()

load_dotenv(override=True)

API_KEY = os.getenv("OPEN_AI_API_KEY")

client = OpenAI(api_key=API_KEY)

csv_path = "files/cleaned_questions.csv"
df = pd.read_csv(csv_path)

topics = [
    "Mathematics",
    "Computer Architecture & Organization",
    "Digital Logic & Circuits",
    "Data Structures & Algorithms",
    "Data Representation & Compression",
    "Operating Systems & System Administration",
    "Database Management Systems",
    "Computer Networks & Communication",
    "Security",
    "Error Handling", 
    "Signal Processing",
    "Data Validation"
]

def classify_question(question: str, topics: List[str]) -> str:
    system_prompt = f"""
        You are a classification assistant.

        Here are the possible topics you must classify into:
        {', '.join(topics)}

        Important rules:
        - Pick exactly one topic from the list.
        - No extra words, no punctuation, no formatting.
        - Output only the topic name as provided.
    """

    prompt = f"""
      Classify the question:
      {question}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.0,
        max_tokens=15  # Limits output length
    )

    # Clean any residual formatting
    topic = response.choices[0].message.content.strip()
    topic = topic.replace('*', '').replace('#', '').replace('-', '').strip()
    topic = topic.split('\n')[0]  # Take only first line if multiple
    return topic



# Enhanced classification function with error handling
def safe_classify(question):
    try:
        return classify_question(question, topics=topics)
    except Exception as e:
        print(f"Error processing question: '{question[:30]}...' - {str(e)}")
        return f"ERROR: {str(e)}"  # Or return None if preferred

# Apply with progress bar and error handling
print("Starting classification process...")
df['classification'] = df['question'].progress_apply(safe_classify)

# Check error rate
error_rate = (df['classification'].str.contains('ERROR').mean() * 100)
print(f"\nClassification completed with {error_rate:.1f}% error rate")


# saving as csv file
output_folder = "files"
output_filename = 'classified_questions.csv'

# Create the folder if it doesn't exist
full_path = os.path.join(output_folder, output_filename)
os.makedirs(output_folder, exist_ok=True)

df.to_csv(full_path, index=False)
# Full path
print(f"\nSaved results to {output_filename}")