from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv(override=True)

API_KEY = os.getenv("OPEN_AI_API_KEY")
# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

topics = [
    {"topic": "Mathematics & Logical Reasoning", "description": "Covers mathematical principles and logical problem-solving techniques, including algebra, geometry, statistics, and analytical reasoning. Keywords: logical reasoning, mathematical principles, algebra, geometry, statistics, analytical reasoning, problem-solving, number theory, calculus, combinatorics, set theory, probability, equations."},
    
    {"topic": "Computer Architecture & Digital Logic", "description": "Focuses on the internal structure of computers, including processors, memory, binary logic, digital circuits, and system-level design. Keywords: computer architecture, processors, memory, binary logic, digital circuits, system design, microarchitecture, CPU, cache memory, register, flip-flops, ALU (Arithmetic Logic Unit), digital design, logic gates, system-level design."},
    
    {"topic": "Data Structures & Algorithms", "description": "Study of efficient methods for organizing, storing, and processing data using structures like arrays, trees, and graphs, along with sorting, searching, and optimization algorithms. Keywords: data structures, algorithms, sorting, searching, optimization, arrays, linked lists, trees, graphs, heaps, queues, stacks, recursion, dynamic programming, algorithm analysis, time complexity."},
    
    {"topic": "Data Representation & Storage", "description": "Methods for representing, storing, and compressing data in digital systems, including number systems, encoding schemes, and data compression techniques. Keywords: data representation, number systems, binary, encoding schemes, compression techniques, data storage, file formats, data encoding, lossless compression, storage devices, bitwise operations, ASCII, Huffman encoding."},
    
    {"topic": "Operating Systems & System Management", "description": "Principles and design of operating systems, including memory management, process scheduling, file systems, and administrative system tasks. Keywords: operating systems, system management, memory management, process scheduling, file systems, kernel, multitasking, system calls, virtual memory, deadlock, semaphore, OS design, resource allocation, device management."},
    
    {"topic": "Database Systems", "description": "Management of structured data using database systems, covering concepts like SQL, NoSQL, data modeling, and transaction management. Keywords: database systems, SQL, NoSQL, data modeling, transaction management, relational databases, database design, normalization, query optimization, indexing, ACID properties, database architecture, schema design, data integrity, ER diagrams."},
    
    {"topic": "Computer Networks & Security", "description": "Study of communication between systems and the protection of networks, covering protocols, cybersecurity principles, encryption, and network architectures. Keywords: computer networks, networking protocols, cybersecurity, encryption, network security, firewalls, TCP/IP, VPNs, routers, DNS, wireless networks, intrusion detection, cryptography, network topologies, data security."},
    
    {"topic": "Software Reliability & Error Handling", "description": "Ensuring software stability through robust error detection, handling, data validation, and system recovery practices. Keywords: software reliability, error handling, exception handling, fault tolerance, robustness, system recovery, data validation, debugging, testing, error prevention, failure detection, logging, error messages, reliability engineering."},
    
    {"topic": "Signal Processing", "description": "Techniques for analyzing, transforming, and extracting information from signals such as audio, images, and sensor data. Keywords: signal processing, audio signals, image processing, sensor data, Fourier transform, filtering, noise reduction, data analysis, wavelets, modulation, digital signals, signal reconstruction, spectral analysis, sampling theorem."}
]


# Get embeddings for the topics' descriptions
def get_topic_embeddings(topics):
    embeddings = []
    for topic in topics:
        response = client.embeddings.create(input=topic["description"], model="text-embedding-ada-002")
        embedding = response.data[0].embedding  # Correct way to access embedding
        embeddings.append(np.array(embedding))
    return np.array(embeddings)

# Get embedding for the question
def get_question_embedding(question: str):
    response = client.embeddings.create(input=question, model="text-embedding-ada-002")
    embedding = response.data[0].embedding  # Correct way to access embedding
    return np.array(embedding)  # Modify based on the actual structure


# Classify the question by finding the most similar topic
def classify_question_with_embeddings(question: str, topics: list):
    topic_embeddings = get_topic_embeddings(topics)
    question_embedding = get_question_embedding(question)
    
    # Compute cosine similarity between the question embedding and each topic embedding
    similarities = cosine_similarity([question_embedding], topic_embeddings)[0]  # Flatten to 1D array

    top_indices = np.argsort(similarities)[::-1][:3]

    top_topics = [topics[idx] for idx in top_indices] 

    
    print(f"\nQuestion:{question}")
    print("\nTop 3 Topics:")
    for idx in top_indices:
        print(f"- {topics[idx]['topic']}: {similarities[idx]:.4f}")
    
    return top_topics


# Example question
question = "Which of the following is a method for embedding a malicious java script code in the content sent to a victims web browser from a vulnerable website?"
# Classify the question
top_topics = classify_question_with_embeddings(question, topics)



# todo add llm reasoning
def select_most_reasonable_topic_with_llm(question, top_topics):
    reasoning_prompt = f"""
      Question: {question}

      Top 3 topics:
      1. {top_topics[0]["topic"]}: {top_topics[0]["description"]}
      2. {top_topics[1]["topic"]}: {top_topics[1]["description"]}
      3. {top_topics[2]["topic"]}: {top_topics[2]["description"]}

      Pick the best matching topic.
      Briefly explain your choice in 1-2 sentences.
      Format:
      Reason: <reason>
      Final: <topic title>
          """
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[
            {"role": "system", "content": "You select the best fitting topic with brief reasoning."},
            {"role": "user", "content": reasoning_prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()



# Now use the LLM to select the final most reasonable topic
final_topic = select_most_reasonable_topic_with_llm(question, top_topics)
print(f"\nSelected Topic (after reasoning):\n{final_topic}")