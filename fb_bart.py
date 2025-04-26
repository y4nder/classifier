from transformers import pipeline

# Load the zero-shot classification pipeline with BART model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the sequence to classify
sequence_to_classify = "From a Score table in SQL in database, the average score for all subjects is to be calculated for each student, and the student number and average score for students with an average score of 80 or higher are to be determined. Which of the following is the appropriate term or phrase to be entered in blank A? Here, a solid underline represents a primary key."

# Define the candidate labels (the classes you want to classify the sequence into)
candidate_labels = [
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

# Perform zero-shot classification
output = classifier(sequence_to_classify, candidate_labels)

# Display the output
print("\nQuestion:")
print(output['sequence'])

print("\nTop Predictions:")
for label, score in zip(output['labels'], output['scores']):
    print(f"- {label}: {score:.4f}")

