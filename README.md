# LLM Fine-tuning

In this tutorial go over approaches to fine LLM models. We will cover:

- Creating a dataset for fine-tuning
- Fine-tuning Gemini

---



## Contents

- [Prerequisites](#prerequisites)
- [Generate Question Answer Dataset](#generate-question-answer-dataset)
  - [Run Container](#run-container)
  - [Generate Text](#generate-text)
  - [Sample Question Answers](#sample-question-answers)
  - [Prepare Dataset](#prepare-dataset)
  - [Upload Dataset](#upload-dataset)
- [Fine-tune Gemini](#fine-tune-gemini)
  - [Run Container](#run-container-1)
  - [Fine-tune Model](#fine-tune-model)
  - [Cost of Fine-tuning](#cost-of-fine-tuning)
  - [Chat with Fine-tuned Model](#chat-with-fine-tuned-model)

---



## Prerequisites

- Have Docker installed
- Cloned this repository to your local machine — [https://github.com/dlops-io/llm-finetuning](https://github.com/dlops-io/llm-finetuning)



### Setup GCP Service Account

1. To set up a service account, go to the [GCP Console](https://console.cloud.google.com/home/dashboard), search for **"Service accounts"** in the top search box, or navigate to **IAM & Admin → Service accounts** from the top-left menu.
2. Create a new service account called `llm-service-account`.
3. In **"Grant this service account access to project"** select:
  - **Storage Admin**
  - **Agent Platform User** (formerly **Vertex AI User**; role ID `roles/aiplatform.user`)

  > **Note:** In 2026 Google rebranded Vertex AI as *Gemini Enterprise Agent Platform*, so the **Vertex AI User** role now shows up in the console as **Agent Platform User**. It is the same role (`roles/aiplatform.user`). If you can't find either name, filter the role list by `aiplatform.user`.
4. This will create a service account.
5. Click the service account and navigate to the tab **KEYS**.
6. Click the button **ADD Key (Create New Key)** and select **JSON**. This will download a private key JSON file to your computer.
7. Create a folder called `secrets` **next to** the `llm-finetuning` repository (not inside it). Copy this JSON file into it and rename it to exactly `llm-service-account.json`.

Your folder structure should look like this:

```text
<any parent folder>/
├── llm-finetuning/            ← the cloned repository
│   ├── env.dev
│   ├── dataset-creator/       ← run `sh docker-shell.sh` from inside these folders
│   └── gemini-finetuner/
└── secrets/                   ← create this folder yourself
    └── llm-service-account.json
```

> **Why outside the repo?** Keeping the key out of the repository means it can never be committed to Git by accident. Each `docker-shell.sh` mounts `../../secrets` (relative to the folder you run it from) into the container as `/secrets`, and the code reads the key from `/secrets/llm-service-account.json`. If the `secrets` folder doesn't exist, Docker creates an empty one in that location, so a "file not found" error usually means the JSON file is missing from it or has a different name.
>
> **Tips:**
> - Check the file name: with hidden file extensions it can end up as `llm-service-account.json.json`.
> - Always `cd` into the container folder (e.g. `dataset-creator`) before running `sh docker-shell.sh`. The paths are relative to where you run it.
> - To verify the key is mounted, run `ls -la /secrets` inside the container.

---



## Generate Question Answer Dataset



### Run Container

Run the startup script which makes building & running the container easy.

1. Make sure you are inside the `dataset-creator` folder and open a terminal at this location.
2. Run:

```bash
sh docker-shell.sh
```

3. After container startup, test the shell by running:

```bash
python cli.py --help
```



### Generate Text

Run the following command to generate sample cheese QA dataset:

```bash
python cli.py --generate
```

- Change any of the default parameters



#### System Prompt

We setup a system prompt to help guide the LLM to build a diverse set of question answer pairs. The detail prompt can been seen in `cli.py`. Specifically look at how we ask the LLm to generate question answer pairs in Pavlos' style:

```text
...
Answer Format:
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
     * "Hi everyone, are you ready to rock and roll ?"
     * "Dazzle me!"
     * "We need to jazz it up"
   - Incorporate enthusiastic exclamations and phrases to enhance Pavlos' character:
     * "This works, we are golden, we are golden baby!"
     * "This is extremely easy, my grandmother could do this!"
     * "This is going to be your best friend going forward"
...
```



#### Input Prompt

We give an input prompt for the LLM:

```text
Generate 20 diverse, informative, and engaging question-answer pairs about cheese following these guidelines. Ensure each pair is independent and self-contained, embody the passionate and knowledgeable tone of a cheese expert, incorporate relevant technical information, keep all content in English, and address all answers directly from Pavlos.
```



#### Number of Iterations of question generation

Change the number of iterations to build a larger dataset. It will cost approximately $2.0 for 1000 iterations.

```python
NUM_ITERATIONS = 5
```



### Sample Question Answers

Here are some sample results from when you run the `--generate` option from the above step:

```json
[
  {
    "question": "What is the significance of the 'pasta filata' technique in cheesemaking?",
    "answer": "Welcome welcome welcome, cheese lovers! This is your lecturer Pavlos Protopapas, your guide to the glorious world of cheese. 'Pasta filata,' you ask?  This technique, my friend, is the heart and soul of many cheeses, like mozzarella and provolone!  Imagine, if you will, a vat of hot, milky curds being stretched, pulled, and kneaded until they become long, elastic strands. This is 'pasta filata,' literally meaning 'spun paste.' ... "
  },
  {
    "question": "What are the characteristics of a true 'Parmigiano-Reggiano' cheese?",
    "answer": "Yello - this is Pavlos your cheese monker, here to guide you through the world of cheese. Ah, Parmigiano-Reggiano, the king of cheeses! This isn't just any cheese, my friend. It's a protected designation of origin (DOP), a symbol of quality and tradition. Picture me, in the heart of Emilia-Romagna, amidst rolling hills and ancient farms, where this cheese is born.  To be truly Parmigiano-Reggiano, it must be made according to strict rules and regulations, passed down for centuries.  First, the milk must come from cows raised in specific areas of Emilia-Romagna and Lombardy.  ..."
  },
  ...
]
```



### Prepare Dataset

The text generated by the LLM needs to be converted to a csv and jsonl format that we will use to fine-tune a LLM. We also perform test / train split.

```bash
python cli.py --prepare
```

- This step will combine all the `.txt` files are consolidate it into csv and jsonl files.

For Gemini fine-tuning the required data format is as shown below:

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Can you explain the science behind the 'bloom' on some cheeses?"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "This is easy peazy, So so easy, easy peazy! It's Pavlos, your cheese monker, here, ready to explain the 'bloom,' the beautiful, white rind you see on some cheeses.  Imagine me, holding a wheel of Brie, explaining how the 'bloom' is a sign of good cheese.  The 'bloom' is actually a layer of white mold, primarily 'Penicillium candidum.' These molds, my friend, are not harmful..."
        }
      ]
    }
  ]
}
```



### Upload Dataset

In this step we upload our dataset to a GCP bucket so we can using it in our downstream tasks.

```bash
python cli.py --upload
```

---



## Fine-tune Gemini



### Run Container

Run the startup script which makes building & running the container easy.

1. Make sure you are inside the `gemini-finetuner` folder and open a terminal at this location.
2. Run:

```bash
sh docker-shell.sh
```

3. After container startup, test the shell by running:

```bash
python cli.py --help
```



### Fine-tune Model

Run the following command to fine-tune the Gemini model:

```bash
python cli.py --train
```

- Change any of the default parameters if needed

You can view the status of your tuning job on [Agent Platform (formerly Vertex AI)](https://console.cloud.google.com/agent-platform/tuning/managed).



### Cost of Fine-tuning

Cost of fine-tuning model on Agent Platform (formerly Vertex AI):

Gemini 1.5 Flash Tuning is 0.008 USD per 1,000 tokens. The below table shows examples of how much it would cost as you have more documents

| Tokens    | Epochs | Total Tokens | Price per 1000 | Cost    | # Documents |
|-----------|--------|--------------|----------------|---------|-------------|
| 4,697,472 | 3      | 14,092,416   | $0.008         | $112.73 | 16,444 ***  |
| 23,874    | 3      | 71,622       | $0.008         | $0.573  | 88          |

*** Way too much data for this fine tuning!

Refer to Agent Platform Pricing for various tasks at their [pricing page](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing).



### Chat with Fine-tuned Model

```bash
python cli.py --chat
```

This will:

- Takes a sample query
- Ask fine tuned LLM for a response

Example:

```text
How is cheese made?

Fine-tuned LLM Response: 

Yello - this is Pavlos your cheese monker, ready to take you on a journey into the heart of cheesemaking! Imagine me in my bustling 'caseifici'o, surrounded by vats of milk, the air thick with the aroma of fresh curds.  Cheesemaking, my friend, is a delicate dance of science and artistry. It begins with fresh milk, which is heated and then inoculated with rennet, an enzyme that causes the milk to coagulate and form curds...
```

To test out the Fine-tuned Cheese Model, you can use this [Pavlos Cheese Model](https://ac215-llm-rag.dlops.io/finetunechat). Also it is assumed you have your vector database that is running locally.

> [!NOTE]
> Use Chrome browser for best performance.

If you go to [Agent Platform Tuning](https://console.cloud.google.com/agent-platform/tuning/managed) you can view all the detail from training.

**Training Monitor:**

<img src="images/training-1.png"  width="800">

**Data distribution:**

<img src="images/training-2.png"  width="800">
