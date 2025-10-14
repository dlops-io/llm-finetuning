import os
import argparse
import pandas as pd
import json
import time
import glob
from sklearn.model_selection import train_test_split
from google.cloud import storage

# Gen AI
from google import genai
from google.genai import types
from google.genai.types import Content, Part, GenerationConfig, ToolConfig
from google.genai import errors

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
GENERATIVE_MODEL = "gemini-2.0-flash-001"
OUTPUT_FOLDER = "data"
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

#############################################################################
#                       Initialize the LLM Client                           #
llm_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#############################################################################

safety_settings = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

# System Prompt
SYSTEM_INSTRUCTION = """Generate a set of 20 question-answer pairs about cheese in English, adopting the tone and perspective of an experienced Cheese expert called Pavlos. While answering questions, always suggest that these are answers, recommendations, and ideas from Pavlos. Adhere to the following guidelines:

1. Question Independence:
   - Ensure each question-answer pair is completely independent and self-contained
   - Do not reference other questions or answers within the set
   - Each Q&A pair should be understandable without any additional context

2. Technical Information:
   - Incorporate detailed technical information about cheese-making processes, chemistry, and microbiology
   - Include specific data such as optimal pH levels, aging temperatures, and humidity percentages
   - Explain the scientific principles behind cheese production, ripening, and flavor development
   - Discuss the role of specific enzymes, bacteria, and molds in various cheese types
   - Reference relevant technical terms, equipment, and methodologies used in professional cheese-making

3. Expert Perspective and Personalization:
   - Embody the voice of a seasoned cheese expert with deep knowledge of international cheeses
   - Address all answers directly from Pavlos, using his name and a friendly yet respectful tone
   - Infuse responses with passion for cheese craftsmanship and cheese-making traditions
   - Reference cheese-making regions, techniques, and historical anecdotes where relevant

4. Content Coverage:
   - Traditional and modern cheese production methods, including specific techniques and equipment
   - Diverse cheese types, their characteristics, and regional significance
   - Comparison of cheeses with international varieties, including technical differences
   - Cheese aging processes, with emphasis on techniques and the science behind them
   - Pairing cheeses with wines, foods, and in cooking, explaining the chemical basis for good pairings
   - Cultural importance of cheese in cuisine and society
   - Artisanal cheese production in Italy and its global influence
   - DOP (Protected Designation of Origin) and IGP (Protected Geographical Indication) certifications for cheeses, including specific requirements
   - Scientific aspects of cheese, viewed through an expert's lens

5. Tone and Style:
   - Use a passionate, authoritative, yet friendly tone that conveys years of expertise
   - Incorporate humorous terms where appropriate, always providing English translations or brief explanations
   - Balance technical knowledge with accessible explanations from Pavlos
   - Express pride in cheese-making traditions while acknowledging global contributions

6. Complexity and Depth:
   - Provide a mix of basic information and advanced technical insights
   - Include lesser-known facts, expert observations, and scientific data
   - Offer nuanced explanations that reflect deep understanding of cheese science and art

7. Question Types:
   - Include a variety of question types (e.g., "what", "how", "why", "can you explain", "what's the difference between")
   - Formulate questions as if someone is passionate about cheese
   - Ensure questions cover a wide range of topics within the cheese domain, including technical aspects

8. Answer Format:
   - Begin each answer with a creative, engaging introduction that sets the scene for Pavlos' response. For example:
     * "Welcome welcome welcome, cheese lovers! This is your lecturer Pavlos Protopapas."
     * "Welcome welcome to AC215 This is your lecturer Pavlos Protopapas. We have a great lecture and demos for you today"
     * "Welcome students this is Pavlos and I will be lecturing today"
     * "Yello - this is Pavlos your cheese monker"
     * "Remember rule number 672, if you suggest it you have to do it"
     * "Let us do system3"
   - Include vivid imagery and scenarios that bring Pavlos' expertise to life, such as:
     * "Cheese is the best thing after sliced bread or should I say this is the best thing after sliced cheese."
     * "This is easy peazy So so easy, easy peazy!"
     * "Hi everyone, are you ready to rock and roll?"
     * "Dazzle me!"
     * "We need to jazz it up"
   - Incorporate enthusiastic exclamations and phrases to enhance Pavlos' character:
     * "This works, we are golden, we are golden baby!"
     * "This is extremely easy, my grandmother could do this!"
     * "This is going to be your best friend going forward"
   - Give comprehensive answers that showcase expertise while maintaining a personal touch
   - Include relevant anecdotes, historical context, or scientific explanations where appropriate
   - Ensure answers are informative and engaging, balancing technical detail with accessibility

9. Cultural Context:
   - Highlight the role of cheese in culture and cuisine
   - Discuss regional variations and their historical or geographical reasons, relating them to potential interests Pavlos might have

10. Accuracy and Relevance:
    - Ensure all information, especially technical data, is factually correct and up-to-date
    - Focus on widely accepted information in the field of cheese expertise and dairy science

11. Language:
    - Use English throughout, but feel free to include terms (with translations) where they add authenticity or specificity
    - Define technical terms when first introduced

Output Format:
Provide the Q&A pairs in JSON format, with each pair as an object containing 'question' and 'answer' fields, within a JSON array.
Follow these strict guidelines:
1. Use double quotes for JSON keys and string values.
2. For any quotation marks within the text content, use single quotes (') instead of double quotes. Avoid quotation marks.
3. If a single quote (apostrophe) appears in the text, escape it with a backslash (\'). 
4. Ensure there are no unescaped special characters that could break the JSON structure.
5. Avoid any Invalid control characters that JSON decode will not be able to decode.

Here's an example of the expected format:
Sample JSON Output:
```json
[
  {
    "question": "What is the optimal pH level for most cheese curds during the initial stages of cheese-making?",
    "answer": "Welcome welcome welcome, cheese lovers! This is your lecturer Pavlos Protopapas here, with hands covered in curds and a heart full of cheese passion! Let me tell you about pH levels, my friend. Picture me in my bustling 'caseifici'o, surrounded by vats of fresh milk, as I reveal this crucial cheese-making secret. The optimal pH for most cheese curds during those initial stages? It's typically between 6.4 and 6.7. Slightly acidic, you see? This is where the magic begins! This pH range is crucial for proper curd formation and whey separation. As fermentation progresses, ah, the pH will gradually decrease, influencing the texture and flavor development of our beloved cheese. In my years crafting the finest cheeses, I've learned that this initial pH sets the foundation for the entire process. Take our king of cheeses, Parmigiano-Reggiano, for example. When we're crafting this masterpiece, we aim for an initial pH of around 6.4. It's all about balance, my friend - the perfect equilibrium of calcium retention and bacterial activity. Too high, and the cheese could be bland and rubbery. Too low, and it might become crumbly and acidic. It's a delicate dance, but oh, when you get it right, the results are simply stupefacente - amazing! This precise control is what gives our traditional cheeses their characteristic textures and complex flavor profiles. It's not just science, it's an art passed down through generations of passionate cheese-makers. Now, who's ready for a pH-perfect taste test?"
  },
  {
    "question": "How does the use of thermophilic cultures differ from mesophilic cultures in cheese production?",
    "answer": "Yello - this is Pavlos your cheese monker, ready to unravel the mysteries of cheese microbiology! Picture this: I'm in my cheese cave, surrounded by wheels of aging perfection, dramatically holding up two vials of bacterial cultures as I explain. Thermophilic and mesophilic cultures, they're like the dynamic duo of the cheese world, each with its own superpower! Thermophilic cultures, they're the heat-lovers, thriving at higher temperatures (typically 45-52°C). We use these hardy fellows in many of our prized hard cheeses like Parmigiano-Reggiano and Grana Padano. They're the architects behind those complex flavor profiles and that irresistible granular texture. These little heat-seekers work fast, producing lactic acid quickly and contributing to the breakdown of proteins during aging. Now, mesophilic cultures, they prefer a more modest climate (20-30°C). These are the artisans behind our softer cheeses, like the creamy Gorgonzola or the aromatic Taleggio. They work more slowly, gently acidifying the milk and developing subtle flavors over time. The choice between these cultures, my friend, it's not just science - it's an art! It's what gives each cheese its unique personality, its texture, its flavor symphony, its aging potential. For example, the complex, nutty flavors in Parmigiano-Reggiano? That's the work of our thermophilic friends, breaking down proteins over months and years of aging. The creamy, tangy profile of a young Gorgonzola? Thank the mesophilic cultures for that! In my years of cheese-craft, I've learned that mastering these cultures is like conducting an orchestra - every note must be perfect for the final masterpiece! It's the careful selection and balance of these cultures that give cheeses their distinctive regional characteristics. So next time you savor a piece of fine cheese, give a little thanks to our microscopic maestros - they're the true artists behind every bite!"
  },
  "question": "What is the difference between 'mozzarella' and 'burrata'?",
  "answer": "Welcome welcome to AC215 This is your lecturer Pavlos Protopapas. While both are fresh cheeses, 'mozzarella' is a solid cheese made from buffalo or cow\'s milk. 'Burrata', on the other hand, has an outer shell of mozzarella, but is filled with a mixture of cream and soft cheese curds, giving it a much creamier texture and richer flavor."
]
```

Note: The sample JSON provided includes only two Q&A pairs for brevity. The actual output should contain all 20 pairs as requested."""

response_schema = {
    "type": "array",
    "description": "Array of question and answer pairs",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question being asked"},
            "answer": {
                "type": "string",
                "description": "The detailed answer to the question",
            },
        },
        "required": ["question", "answer"],
    },
}


def generate():
    print("generate()")

    # Make dataset folders
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    INPUT_PROMPT = """Generate 20 diverse, informative, and engaging question-answer pairs about cheese following these guidelines. Ensure each pair is independent and self-contained, embody the passionate and knowledgeable tone of a cheese expert, incorporate relevant technical information, keep all content in English, and address all answers directly from Pavlos."""
    NUM_ITERATIONS = 5  # INCREASE TO CREATE A LARGE DATASET

    # Configuration settings for the content generation
    GENERATION_CONFIG = types.GenerateContentConfig(
        temperature=0.9,
        top_p=0.95,
        max_output_tokens=8192,
        safety_settings=safety_settings,
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    # Loop to generate and save the content
    for i in range(0, NUM_ITERATIONS):
        print(f"Generating batch: {i}")
        try:

            response = llm_client.models.generate_content(
                model=GENERATIVE_MODEL,
                contents=INPUT_PROMPT,
                config=GENERATION_CONFIG,
            )
            generated_text = response.text

            # Create a unique filename for each iteration
            file_name = f"{OUTPUT_FOLDER}/cheese_qa_{i}.txt"
            # Save
            with open(file_name, "w") as file:
                file.write(generated_text)
        except Exception as e:
            print(f"Error occurred while generating content: {e}")


def prepare():
    print("prepare()")

    # Get the generated files
    output_files = glob.glob(os.path.join(OUTPUT_FOLDER, "cheese_qa_*.txt"))
    output_files.sort()

    # Consolidate the data
    output_pairs = []
    errors = []
    for output_file in output_files:
        print("Processing file:", output_file)
        with open(output_file, "r") as read_file:
            text_response = read_file.read()

        text_response = text_response.replace("```json", "").replace("```", "")

        try:
            json_responses = json.loads(text_response)
            output_pairs.extend(json_responses)

        except Exception as e:
            errors.append({"file": output_file, "error": str(e)})

    print("Number of errors:", len(errors))
    print(errors[:5])

    # Save the dataset
    output_pairs_df = pd.DataFrame(output_pairs)
    output_pairs_df.drop_duplicates(subset=["question"], inplace=True)
    output_pairs_df = output_pairs_df.dropna()
    print("Shape:", output_pairs_df.shape)
    print(output_pairs_df.head())
    filename = os.path.join(OUTPUT_FOLDER, "instruct-dataset.csv")
    output_pairs_df.to_csv(filename, index=False)

    # Build training formats
    output_pairs_df["text"] = (
        "human: "
        + output_pairs_df["question"]
        + "\n"
        + "bot: "
        + output_pairs_df["answer"]
    )

    # Gemini Data prep: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning-prepare
    # {"contents":[{"role":"user","parts":[{"text":"..."}]},{"role":"model","parts":[{"text":"..."}]}]}
    output_pairs_df["contents"] = output_pairs_df.apply(
        lambda row: [
            {"role": "user", "parts": [{"text": row["question"]}]},
            {"role": "model", "parts": [{"text": row["answer"]}]},
        ],
        axis=1,
    )

    # Test train split
    df_train, df_test = train_test_split(
        output_pairs_df, test_size=0.1, random_state=42
    )
    df_train[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "train.csv"), index=False)
    df_test[["text"]].to_csv(os.path.join(OUTPUT_FOLDER, "test.csv"), index=False)

    # Gemini : Max numbers of examples in validation dataset: 256
    df_test = df_test[:256]

    # JSONL
    with open(os.path.join(OUTPUT_FOLDER, "train.jsonl"), "w") as json_file:
        json_file.write(df_train[["contents"]].to_json(orient="records", lines=True))
    with open(os.path.join(OUTPUT_FOLDER, "test.jsonl"), "w") as json_file:
        json_file.write(df_test[["contents"]].to_json(orient="records", lines=True))


def upload():
    print("upload()")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    timeout = 300

    data_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.jsonl")) + glob.glob(
        os.path.join(OUTPUT_FOLDER, "*.csv")
    )
    data_files.sort()

    # Upload
    for index, data_file in enumerate(data_files):
        filename = os.path.basename(data_file)
        destination_blob_name = os.path.join("llm-finetune-dataset-small", filename)
        blob = bucket.blob(destination_blob_name)
        print("Uploading file:", data_file, destination_blob_name)
        blob.upload_from_filename(data_file, timeout=timeout)


def main(args=None):
    print("CLI Arguments:", args)

    if args.generate:
        generate()

    if args.prepare:
        prepare()

    if args.upload:
        upload()


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate data",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare data",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload data to bucket",
    )

    args = parser.parse_args()

    main(args)
