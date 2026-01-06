QA Generator

QA Generator is  an application for generating quality **question–answer pairs** from documents.
It combines **question generation (QG)** and **extractive question answering (QA)** models trained on **SQuAD v2**, and provides an interactive **Streamlit** interface.

The application supports multiple model backends for both question generation and answer extraction.

---

## Features

* Upload documents in **PPTX, DOCX, or TXT** format
* Automatic text cleaning and paragraph segmentation
* Question generation using:

  * T5 (fine-tuned)
  * FLAN-T5 (instruction-tuned + fine-tuned)
* Answer extraction using:

  * DeBERTa-v3
  * DistilBERT
* Heuristic filtering for question and answer quality
* Interactive Streamlit UI
* Fully local inference (no external APIs)

---

## System Overview

Document
↓
Text Cleaning & Paragraph Splitting
↓
Question Generation (T5 / FLAN-T5)
↓
Question Quality Filtering
↓
Answer Extraction (DeBERTa / DistilBERT)
↓
Answer Quality Filtering
↓
Final Question–Answer Pairs



## Models

### Question Generation (Seq2Seq)

T5

* Base checkpoint: t5-base
* Fine-tuned on SQuAD v2 using context and questions for the given context.
* Task: context → question

FLAN-T5

* Base checkpoint: google/flan-t5-base
* Fine-tuned on SQuAD v2 using context and questions for the given context.
* Task: context → question

Both models are trained using the same supervised learning setup.
The difference lies in the pretrained checkpoint (FLAN-T5 is instruction-tuned).

---

### Question Answering (Extractive QA)

DeBERTa

* Base checkpoint: microsoft/deberta-v3-base
* Fine-tuned on SQuAD v2 context, questions and answers for the given answer.

DistilBERT

* Base checkpoint: distilbert-base-uncased
* Fine-tuned on SQuAD v2 context, questions and answers for the given answer.

Both QA models learn to predict answer spans:

(question, context) → (start token, end token)

---


## Notes

* FLAN-T5 performs better with instruction-style prompts
* DeBERTa provides higher accuracy; DistilBERT provides faster inference
* Designed for exam preparation, study material generation, and document understanding

# Example

![Screenshot 1](images/Output%201.jpg)
![Screenshot 2](images/output%202.jpg)
![Screenshot 3](images/output%203.jpg)
![Screenshot 4](images/output%204.jpg)
![Screenshot 5](images/output%205.jpg)
![Screenshot 6](images/output%206.jpg)



