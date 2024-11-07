# Workload queries
This is the dataset used for running workload on Nano Jetson developer kit 4GB board.

## Queries for workload

To effectively run our experiments, we curated a dataset comprising 100 queries for each of the five categories. These queries were selected from established NLP datasets commonly used for large language models (LLMs) and aligned with the specific category types.

**Simple Queries**:

- **Dataset**: WikiQA, a question-answering corpus from Microsoft ([*Wiki_qa*](https://huggingface.co/datasets/microsoft/wiki_qa)).
    - Example Queries:
        - "How does interlibrary loan work?"
        - "Where is money made in the United States?"
        - "How does a dim sum restaurant work?"

**Complex Queries**:

- **Dataset**: SQuAD v2.0 (Dev set), a benchmark dataset for machine comprehension of text ([*The Stanford Question Answering Dataset*](https://rajpurkar.github.io/SQuAD-explorer/)).
    - Example Queries:
        - "What does the private education student financial assistance help current high school students who are turned away do?"
        - "Whose infected corpse was one of the ones catapulted over the walls of Kaffa by the Mongol army?"

**Conversational Queries**:

- **Dataset**: Friends Dataset, containing speech-based dialogue from the Friends TV sitcom, extracted from the SocialNLP EmotionX 2019 challenge ([Michelle Li](https://huggingface.co/datasets/michellejieli/friends_dataset)).
    - Example Queries:
        - "Oh, that's right. It's your first day! So, are you psyched to fight fake crime with your robot sidekick?"
        - "Whoa!! Now look, don’t be just blurting stuff out. I want you to really think about your answers, okay?"

**Task-Oriented Queries**:

- **Dataset**: MultiWOZ 2.2, a multi-domain Wizard-of-Oz dataset for task-oriented dialogue systems.We retrieved only the user utterances from the conversation per task and combined them for the model to infer and give the steps in accomplishment of the task ([salesforce](https://github.com/salesforce/DialogStudio/tree/main/task-oriented-dialogues/MULTIWOZ2_2)).
    - Example Query:
        - "This is a bot helping users to find a restaurant and find an attraction. Given the dialog context, please generate a relevant system response for the user: <USER> Hi, could you help me with some information on a particular attraction? <USER> It is called Nusha. Can you tell me a little bit about it? <USER> Thank you. Please get me information on a particular restaurant called Thanh Binh. <USER> Thanks, please make a reservation there for 6 people at 17:15 on Saturday."

**Contextual Queries**:

- **Dataset**: Validation set from Question Answering in Context (QuAC), a dataset for modeling, understanding, and participating in information-seeking dialogue ([*Question Answering in Context*](https://quac.ai/)).
    - Example Query Structure:
        - ::context:: (text provided) ::question:: "Were they ever in any other TV shows or movies?"