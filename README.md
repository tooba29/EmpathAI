# EmpathAI: Emotion-Aware Chatbot
EmpathAI Chatbot, your empathetic companion that not only listens but understands your feelings. 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [How It Works](#how-it-works)
  - [Emotion Detection](#emotion-detection)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Response Generation & LLM](#response-generation--llm)
  - [Conversation Memory & Analysis](#conversation-memory--analysis)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Overview

EmpathAI is designed to create a truly empathetic conversational experience. The chatbot detects the user's emotions and overall sentiment, adjusts its tone accordingly via a powerful large language model (LLM), and provides not only engaging responses during the conversation but also an emotional summary afterward. Covering anger, disgust, fear, joy, neutral, sadness, and surprise emotions.

## Features

- **Emotion Detection:**  
  Uses the fine-tuned model `j-hartmann/emotion-english-distilroberta-base` to detect key emotions (anger, disgust, fear, joy, neutral, sadness, and surprise) from user inputs.

- **Sentiment Analysis:**  
  Integrates a sentiment analysis pipeline using the `distilbert/distilbert-base-uncased-finetuned-sst-2-english` model to understand the overall positive or negative sentiment of the conversation, further informing the response tone.

- **Response Generation (LLM):**  
  Generates empathetic and contextually relevant responses using the Mistral language model (`mistralai/Mistral-7B-Instruct-v0.1`) through the Hugging Face Inference API. The LLM's tone and style are adapted based on the emotion detection and sentiment analysis results.

- **Conversation Summarization & Emotional Analysis:**  
  Summarizes each turn of the conversation and compiles these summaries to produce a final emotional journey analysis. This analysis, along with an interactive emotion distribution chart, offers insights into the evolution of the user's emotional state.

- **Session Management:**  
  Utilizes Streamlit's session state to keep track of conversation history, emotion counts, sentiment records, and context. This ensures continuity and a personalized experience throughout the user's session.

## Project Structure

- **model.py**  
  Contains core functions, including:
  - `detect_emotion(text)`: Analyzes a message using the emotion detection model to provide the dominant emotion and detailed scores.
  - `generate_summary(previous_input, emotion_scores, bot_output)`: Summarizes a single exchange to maintain conversational context.
  - `generate_response(user_input, sentiment, emotion, previous_context="")`: Crafts empathetic responses by combining user input, detected sentiment, and emotions.
  - `generate_emotion_story(summaries, emotion_scores)`: Provides an analytical reflection of the conversation’s emotional progression using stored summaries and final emotion distribution.

- **agent.py**  
  Acts as a bridge, handling the conversation flow:
  - `get_response_with_emotion_and_sentiment(user_input, sentiment, previous_context="")`:  
    - Calls `detect_emotion` to understand the current emotion.
    - Generates a summary if there’s previous conversation data.
    - Invokes `generate_response` to create the bot's empathetic reply.
    - Updates the conversation memory with previous inputs, emotion scores, and bot outputs.

- **app.py**  
  The main front-end built with Streamlit, responsible for:
  - User onboarding and personalized greetings.
  - Displaying the live chat interface with conversation history.
  - Integrating sentiment and emotion analysis.
  - Visualizing emotion distribution via a Plotly pie chart.
  - Rendering the final emotional journey summary and enabling chat reset.

## Technology Stack

- **Python:** Primary programming language.
- **Streamlit:** For building the interactive web interface.
- **Hugging Face Transformers:** For emotion detection and sentiment analysis.
- **Hugging Face Inference API:** Accesses the Mistral LLM (`mistralai/Mistral-7B-Instruct-v0.1`) for response generation and conversation summarization.
- **Plotly Express & Pandas:** For data visualization and managing conversation-related data.

## How It Works

### Emotion Detection

- **What It Does:**  
  Uses a pre-trained and fine-tuned transformer model (`j-hartmann/emotion-english-distilroberta-base`) to identify the emotional state of the user's input.
  
- **Impact on LLM Tone:**  
  The detected emotion (e.g., joy, sadness, or anger) is passed as a parameter when constructing the prompt for the LLM. This ensures the LLM adapts its tone — for example, a supportive reply when sadness is detected or a calming reply when anger is noted.

### Sentiment Analysis

- **What It Does:**  
  The sentiment analysis is performed using DistilBERT (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`) which classifies the overall sentiment of the text as either positive or negative.
  
- **Purpose & Usage:**  
  The sentiment result provides an overarching tone context. Combined with the granular emotion scores, it helps the LLM craft responses that are not just empathetic but also aligned with the overall positive or negative sentiment of the conversation.

### Response Generation & LLM

- **LLM In Use:**  
  The large language model used is `mistralai/Mistral-7B-Instruct-v0.1`, which is designed to follow instructions and generate human-like text.
  
- **How It Works:**  
  The LLM receives a prompt that includes:
  - The user's current input.
  - The detected sentiment and emotion.
  - Previous conversation context and summaries.
  
  The LLM then generates an empathetic reply that:
  - Uses an appropriate tone based on the detected emotion and sentiment.
  - Incorporates context to make the conversation feel continuous and personalized.
  
- **Example:**  
  If a user expresses sadness, the detection model marks the input as "sadness" and the sentiment analyzer may indicate a negative sentiment. The prompt to the LLM would then instruct it to generate a response that is warm, understanding, and encouraging, thereby adapting its tone to the user's emotional state.

### Conversation Memory & Analysis

- **Tracking Conversation:**  
  Streamlit's session state is used to store conversation history, including each chat turn, emotion counts, sentiment values, and summaries. This data accumulates across the session.
  
- **Summary Generation:**  
  Each chat turn is summarized to maintain context. At the end of the conversation, these summaries are compiled and analyzed by the LLM to produce an overall emotional journey summary, which is then visualized using a Plotly pie chart.

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/tooba29/EmpathAI.git
   cd EmpathAI
2. **Configure Environment Variables:**
   ```bash
   export HF_TOKEN='your_huggingface_api_token'
   ```
   if you're using a .env file (with a package like python-dotenv), create it in the project root and add
   ```bash
   HF_TOKEN=your_huggingface_api_token
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
