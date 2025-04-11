import os
import requests
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

HF_TOKEN = os.getenv("HF_TOKEN")
MISTRAL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# load the fine-tuned model and tokenizer for emotion classification
model_name = "j-hartmann/emotion-english-distilroberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

emotions_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)



# now returns all emotions with their scores and the top emotion 
def detect_emotion(text):
    results = emotions_classifier(text)[0]
    emotion_scores = {entry['label']: entry['score'] for entry in results}
    top_emotion = max(results, key=lambda x: x['score'])['label']
    return top_emotion, emotion_scores


## added feature that generates summmary
def generate_summary(previous_input, emotion_scores, bot_output):
    # Create a summary prompt
    emotion_str = ", ".join(f"{k}: {v:.2f}" for k, v in emotion_scores.items())
    summary_prompt = (
        "Summarize this chat exchange clearly for future context:\n"
        f"User input: {previous_input}\n"
        f"Detected emotions: {emotion_str}\n"
        f"Bot response: {bot_output}\n"
        "Summary:"
    )
    payload = {
        "inputs": summary_prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    generated = response.json()
    full_text = generated[0]["generated_text"]
    return full_text.split("Summary:")[-1].strip()



# generating response
def generate_response(user_input, sentiment, emotion, previous_context=""):
    prompt = (
        previous_context +
        "You are an empathetic assistant who understands both sentiments and emotions.\n"
        f"User said: {user_input}\n"
        f"Detected sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})\n"
        f"Detected emotion: {emotion}\n"
        "Bot:\n"
    )
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    generated = response.json()
    full_text = generated[0]["generated_text"]
    return full_text.split("Bot:")[-1].strip()


## for emotion summary
def generate_emotion_story(summaries, emotion_scores):
    summary_text = "\n".join(f"- {s}" for s in summaries)
    emotion_list = ", ".join(f"{e}: {round(score*100, 2)}%" for e, score in emotion_scores.items())

    prompt = (
        "You are an emotional analyst. Based on the following conversation details, provide a concise, complete analysis in two to three sentences. "
        "Please ensure that the analysis is complete and does not end abruptly.\n\n"
        f"Emotion summaries over time:\n{summary_text}\n\n"
        f"Final emotion distribution: {emotion_list}\n\n"
        "Analysis:\n"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.75,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    generated = response.json()
    full_text = generated[0]["generated_text"]
    return full_text.split("Analysis:")[-1].strip()




