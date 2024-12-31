import time
from transformers import pipeline


def benchmark_model(model_name, command_text):
    print(f"Benchmarking model: {model_name}")
    nlp = pipeline(
        "zero-shot-classification",
        model=model_name,
        tokenizer=model_name,
        framework="pt",
        device=-1,  # Change to 0 if using GPU
    )
    start_time = time.perf_counter()
    result = nlp(
        command_text,
        candidate_labels=[
            "greetings",
            "navigation",
            "application",
            "financial",
            "student_affairs",
            "admission",
            "computer_science",
            "doctor_availability",
            "none",
            "kill",
        ],
        hypothesis_template="This text is about {}.",
        multi_label=True,
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Classification took {elapsed_time:.4f} seconds.")
    print("Result:", result)
    print("-" * 50)


if __name__ == "__main__":
    command = "Hey"
    benchmark_model("facebook/bart-large-mnli", command)
    benchmark_model("valhalla/distilbart-mnli-12-3", command)
