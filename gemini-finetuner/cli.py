import os
import argparse
import time
from google.cloud import storage
from google import genai
from google.genai import types

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
TRAIN_DATASET = "gs://cheese-dataset/llm-finetune-dataset-small/train.jsonl"  # Replace with your dataset
VALIDATION_DATASET = "gs://cheese-dataset/llm-finetune-dataset-small/test.jsonl"  # Replace with your dataset
# gemini-3.1-flash-lite: tune in a single region; serve on multi-region "us"
TUNING_LOCATION = "us-central1"
SERVING_LOCATION = "us"
GENERATIVE_SOURCE_MODEL = "gemini-3.1-flash-lite"

#############################################################################
# Train client: single-region (us-central1) — required to create tuning jobs
train_client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=TUNING_LOCATION,
)

# Chat client: multi-region "us" — required to call tuned Gemini 3.1 endpoints
chat_client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=SERVING_LOCATION,
    http_options=types.HttpOptions(
        base_url="https://aiplatform.us.rep.googleapis.com",
    ),
)
#############################################################################

# Configuration settings for content generation
generation_config = types.GenerateContentConfig(
    max_output_tokens=3000,
    temperature=0.75,
    top_p=0.95,
)


def train(wait_for_job=False):
    print("train()")

    # Create tuning dataset
    training_dataset = types.TuningDataset(gcs_uri=TRAIN_DATASET)
    validation_dataset = types.TuningDataset(gcs_uri=VALIDATION_DATASET)

    # Create tuning job config
    tuning_config = types.CreateTuningJobConfig(
        epoch_count=1,
        learning_rate_multiplier=1.0,
        adapter_size="ADAPTER_SIZE_FOUR",
        tuned_model_display_name="pavlos-cheese-demo-v3.1",
        validation_dataset=validation_dataset,
    )

    # Start supervised fine-tuning
    tuning_job = train_client.tunings.tune(
        base_model=GENERATIVE_SOURCE_MODEL,
        training_dataset=training_dataset,
        config=tuning_config,
    )

    print("Training job started. Monitoring progress...\n")
    print(f"Job name: {tuning_job.name}")

    # Wait and refresh
    time.sleep(60)
    tuning_job = train_client.tunings.get(name=tuning_job.name)

    if wait_for_job:
        print("Check status of tuning job:")
        print(f"State: {tuning_job.state}")

        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        }

        while tuning_job.state not in completed_states:
            time.sleep(60)
            tuning_job = train_client.tunings.get(name=tuning_job.name)
            print(f"Job in progress... State: {tuning_job.state}")

    print(f"\nTuned model info: {tuning_job.tuned_model}")
    if tuning_job.tuned_model:
        print(f"Tuned model name: {tuning_job.tuned_model.model}")
        print(f"Tuned model endpoint: {tuning_job.tuned_model.endpoint}")
        print(
            "For --chat, paste that endpoint into MODEL_ENDPOINT "
            "(expect locations/us/... for gemini-3.1-flash-lite)."
        )
    print(f"Experiment: {tuning_job.experiment}")

    return tuning_job


def chat():
    print("chat()")
    # Get the model endpoint from Vertex AI Tuning UI after the job succeeds.
    # For gemini-3.1-flash-lite serving, use locations/us/endpoints/...
    # https://console.cloud.google.com/vertex-ai/studio/tuning?project=ac215-project
    MODEL_ENDPOINT = "projects/129349313346/locations/us/endpoints/7655908260297310208"

    query = "How is cheese made in Greece? Answer pavlos style?"
    print("query: ", query)
    print("Using serving location:", SERVING_LOCATION)

    response = chat_client.models.generate_content(
        model=MODEL_ENDPOINT,
        contents=query,
        config=generation_config,
    )
    generated_text = response.text
    print("Fine-tuned LLM Response:", generated_text)


def main(args=None):
    print("CLI Arguments:", args)

    if args.train:
        train(wait_for_job=False)

    if args.chat:
        chat()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train model",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with model",
    )

    args = parser.parse_args()

    main(args)
