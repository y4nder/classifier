from transformers import pipeline


# https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
# https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0

# Load the zero-shot-classifier pipeline
zeroshot_classifier = pipeline(
    "zero-shot-classification",
    # model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
    # model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    model="MoritzLaurer/bge-m3-zeroshot-v2.0"
)

# Your input text
text = "ECC is used for error detection and correction in memory. When n+2 redundant bits are required for a data bus having a width of 2n bits, what is the number of redundant bits that are required for a data bus having a width of 128 bits?."
# The list of classes (topics) you want the model to choose from
classes = [
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
    "Data Validation",
    "Logical Reasoning"
]

# The hypothesis template (how the model thinks about the classes)
hypothesis_template = "This example is about {}"

# Classify
output = zeroshot_classifier(
    text,
    classes,
    hypothesis_template=hypothesis_template,
    multi_label=False  # If you want exactly 1 class -> False
)

print("\nQuestion:")
print(output['sequence'])

print("\nTop Predictions:")
for label, score in zip(output['labels'], output['scores']):
    print(f"- {label}: {score:.4f}")

