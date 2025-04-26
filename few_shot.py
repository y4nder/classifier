from setfit import SetFitModel, Trainer

# Download from Hub and run inference
model = SetFitModel.from_pretrained("tyzp-INC/few-shot-multilingual-e5-large-xnli")

train_texts = [
    "What is the integral of sin(x)?",
    "Define TCP/IP layers.",
    "How does an operating system manage memory?",
    "Explain binary trees.",
]

train_labels = [
    "Mathematics",
    "Computer Networks",
    "Operating Systems",
    "Data Structures",
]

trainer = Trainer(
    model=model,
    train_dataset={"text": train_texts, "label": train_labels},
    # loss_class="CosineSimilarityLoss",  # (default for SetFit)
    batch_size=8,
    num_iterations=20,  # Small, few-shot training
)

trainer.train()

# Run inference
preds = model.predict(["How do I calculate the derivative of x^2?", "What is IP address allocation?"])
print(preds)

