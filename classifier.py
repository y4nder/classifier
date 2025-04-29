from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from embeddingStore import EmbeddingStorage
import numpy as np
import os

# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve the OpenAI API key from environment variables
API_KEY = os.getenv("OPEN_AI_API_KEY")

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=API_KEY)

# Initialize the embedding storage for managing topic embeddings
embedding_storage = EmbeddingStorage(openai=client)

# Function to get the embedding for a given question
def get_question_embedding(question: str):
    """
    Generate an embedding vector for a given question using OpenAI's embedding model.

    Args:
        question (str): The question text.

    Returns:
        np.ndarray: A NumPy array representing the embedding vector.
    """

    # Use OpenAI API to generate an embedding for the input question
    response = client.embeddings.create(input=question, model="text-embedding-ada-002")
    embedding = response.data[0].embedding  # Extract the embedding from the response
    return np.array(embedding)  # Convert the embedding to a NumPy array

# Function to classify a question by finding the most similar topic
def classify_question_with_embeddings(question: str, topics: list): 
    """
    Classify a question by finding the most semantically similar topics based on embeddings.

    This function generates an embedding for the given input question, compares it to pre-stored 
    embeddings of topics using cosine similarity, and returns the top 3 topics most similar to the question.

    Args:
        question (str): The input question to classify.
        topics (list): A list of topic names corresponding to the stored topic embeddings.
    Returns:
        list: A list of the top 3 topics (strings) most similar to the input question.
    """
    # Retrieve stored embeddings for topics
    topic_embeddings = embedding_storage.get_embeddings()
    # Generate an embedding for the input question
    question_embedding = get_question_embedding(question)   
    
    # Compute cosine similarity between the question embedding and each topic embedding
    similarities = cosine_similarity([question_embedding], topic_embeddings)[0]  # Flatten to 1D array
    # Get the indices of the top 3 most similar topics
    top_indices = np.argsort(similarities)[::-1][:3]
    # Retrieve the top 3 topics based on similarity
    top_topics = [topics[idx] for idx in top_indices]     
    return top_topics

# Function to select the most reasonable topic using an LLM
def select_most_reasonable_topic_with_llm(question, top_topics):
    """
    Use LLM to select the most appropriate topic among top candidates.

    Args:
        question (str): The question text.
        top_topics (list): A list of top 3 topic dictionaries containing 'topic' and 'description'.

    Returns:
        str: The final selected topic name.
    """
    
    reasoning_prompt = f"""
      Question: {question}

      Top 3 topics:
      1. {top_topics[0]["topic"]}: {top_topics[0]["description"]}
      2. {top_topics[1]["topic"]}: {top_topics[1]["description"]}
      3. {top_topics[2]["topic"]}: {top_topics[2]["description"]}

      Pick the best matching topic.
        Important rules:
        only return the topic name
          """
    
    # Use OpenAI API to get the best matching topic
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": "You select the best fitting topic."},
            {"role": "user", "content": reasoning_prompt}
        ]
    )
    
    # Return the selected topic name
    return response.choices[0].message.content.strip()

# Function to classify the difficulty of a question
def classify_difficulty(question: str) -> str:
    """
    Classify the difficulty of a given question into 'easy', 'average', or 'difficult' using LLM.

    Args:
        question (str): The question text.

    Returns:
        str: The difficulty level ('easy', 'average', 'difficult').
    """
     
    system_prompt = """
        You are a difficulty classification assistant.

        Here are the possible difficulty levels you must classify into:
        easy, average, difficult

        Important rules:
        - Pick exactly one difficulty level from the list.
        - No extra words, no punctuation, no formatting.
        - Output only the difficulty level as provided.
    """

    # Define the user prompt with the question to classify
    prompt = f"""
      Classify the difficulty of the question:
      {question}
    """

    try:
        # Use OpenAI API to classify the difficulty
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.0,  # Ensure deterministic output
            max_tokens=5  # Limits output length
        )

        # Extract and clean the difficulty level from the response
        difficulty = response.choices[0].message.content.strip()
        difficulty = difficulty.replace('*', '').replace('#', '').replace('-', '').strip()
        difficulty = difficulty.split('\n')[0]  # Take only first line if multiple
        return difficulty

    except Exception as e:
        # Log the error and return an error message
        print(f"Error processing difficulty for question: '{question[:30]}...' - {str(e)}")
        return f"ERROR: {str(e)}"  # Or return None if preferred

# Function to extract the final topic for a question
def extract_final_topic(question: str, topics: list) -> str:
    """
    Extract the final most relevant topic for a given question.

    This function first identifies the top 3 most semantically similar topics to the question 
    by comparing embeddings. It then uses a language model to select the single most appropriate 
    topic from the top candidates.

    Args:
        question (str): The input question for which a topic needs to be determined.
        topics (list): A list of available topic names.

    Returns:
        str: The final selected topic name that best matches the question.
    """
    # Get the top 3 most similar topics
    top_topics = classify_question_with_embeddings(question, topics)
    # Use LLM to select the most reasonable topic from the top 3
    final_topic = select_most_reasonable_topic_with_llm(question, top_topics)
    return final_topic
