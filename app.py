import streamlit as st
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
from agent import get_response_with_emotion_and_sentiment 
from transformers import pipeline
from model import generate_emotion_story


st.set_page_config(page_title="Emotion-Aware Chatbot", layout="centered")
st.title("🧠 EmpathAI")
st.markdown("""
**Emotion-Aware Chatbot**

*Welcome to EmpathAI Chatbot, your empathetic companion that not only listens but understands your feelings. By leveraging a fine-tuned emotion detection model alongside a powerful language generator, this chatbot interprets the emotion in your messages — covering anger, disgust, fear, joy, neutral, sadness, and surprise — and replies with caring, personalized responses. Experience an engaging conversation where technology meets empathy.*
""")


st.divider()

GENDER_AVATARS = {
    "Male": "👨",
    "Female": "👩",
    "Other": "🧑"
}

# emotions
emotion_emojis = {
    "anger": "🤬",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😀",
    "neutral": "😐",
    "sadness": "😭",
    "surprise": "😲"
}


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "full_context" not in st.session_state:
    st.session_state["full_context"] = ""
if "user_info" not in st.session_state:
    st.session_state["user_info"] = None
if "last_processed_message" not in st.session_state:
    st.session_state["last_processed_message"] = ""
if "end_chat" not in st.session_state:
    st.session_state["end_chat"] = False
if "emotion_counts" not in st.session_state:
    st.session_state["emotion_counts"] = {
        "anger": 0, "disgust": 0, "fear": 0,
        "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0
    }
if "sentiments" not in st.session_state:
    st.session_state["sentiments"] = []

#conversation memory
if "conversation_memory" not in st.session_state:
    st.session_state["conversation_memory"] = {
        "summaries": [],
        "previous_input": None,
        "previous_emotion_scores": None,
        "previous_bot_output": None,
        "is_first_prompt": True
    }

#loading sentiment classifier
sentiment_classifier = pipeline("sentiment-analysis",
                                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")


# user personalized experience 
if st.session_state["user_info"] is None:
    with st.container():
        st.header("Welcome to the Chat Bot!")
        st.write("Please enter your details below to get started:")
        with st.form(key="onboarding_form"):
            user_name = st.text_input("Enter your name", key="name_input")
            gender = st.selectbox("Select your gender", ["Male", "Female", "Other"], key="gender_select")
            submitted = st.form_submit_button("Start Chat")
        if submitted:
            if user_name.strip() == "":
                st.error("Please enter your name to continue.")
            else:
                st.session_state["user_info"] = {"name": user_name.strip(), "gender": gender}
    st.stop()  


# Display the welcome header user greeting
if not st.session_state.get("end_chat", False):
    st.markdown(
        f"Welcome {GENDER_AVATARS.get(st.session_state['user_info']['gender'], '👤')} **{st.session_state['user_info']['name']}**!"
    )
    st.divider()


conversation_placeholder = st.empty()
typing_placeholder = st.empty()

# function to render the complete conversation

def render_chat():
    conversation_placeholder.empty()
    with conversation_placeholder.container():
        for speaker, message, timestamp in st.session_state["chat_history"]:
            if speaker == "User":
                avatar = GENDER_AVATARS.get(st.session_state["user_info"].get("gender"), "👤")
                display_name = st.session_state["user_info"].get("name", "User")
            else:
                avatar = "🤖"
                display_name = "Bot"
            with st.chat_message(avatar):
                st.markdown(f"**{display_name}**\n*{timestamp}*")
                st.markdown(message)


# chat Interface 
if st.session_state["user_info"] is not None and not st.session_state["end_chat"]:
    # render conversation history
    render_chat()
    st.divider()
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Type your message", key="chat_input",
                                       label_visibility="collapsed", placeholder="Say something...")
            send_clicked = st.form_submit_button("Send")
    
    if send_clicked and user_input.strip() and st.session_state["last_processed_message"] != user_input:
        st.session_state["last_processed_message"] = user_input
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["chat_history"].append(("User", user_input, timestamp))
        
        # run sentiment analysis on user input.
        sentiment_result = sentiment_classifier(user_input)[0]
        st.session_state["sentiments"].append(sentiment_result)
        
        render_chat()  
        
        
        typing_placeholder.markdown("🤖 Bot is typing...")
        time.sleep(2)  # simulate delay
        
        try:
            
            # Get bot response with both sentiment and detected emotion.
            # edited 
            sentiment, emotion, response = get_response_with_emotion_and_sentiment(user_input, sentiment_result, "") # old 
            # sentiment, emotion, response = get_response_with_emotion_and_sentiment(user_input, sentiment_result, st.session_state["conversation_memory"])

            st.session_state["emotion_counts"][emotion.lower()] += 1
            bot_message = f"**Detected Emotion:** {emotion_emojis.get(emotion.lower(), '🤖')} {emotion}\n\n" \
                          f"{response}"
            st.session_state["chat_history"].append(("Bot", bot_message, timestamp))
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        typing_placeholder.empty()  
        render_chat()  
    
    # end chat and clear conversation button
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧹 Clear Conversation"):
                st.session_state["chat_history"] = []
                st.session_state["full_context"] = ""
                st.session_state["last_processed_message"] = ""
                render_chat()
        with col2:
            if st.button("End Chat"):
                st.session_state["end_chat"] = True


# analysis page

if st.session_state["user_info"] is not None and st.session_state["end_chat"]:
    st.header("Chat Analysis")
    user_name = st.session_state["user_info"]["name"]
    st.write(f"{user_name}, here is your emotional analysis based on our conversation:")
    st.subheader("Emotion Distribution")
    
    # build a dataframe from the emotion counts stored in session state
    df = pd.DataFrame({
        "Emotion": list(st.session_state["emotion_counts"].keys()),
        "Count": list(st.session_state["emotion_counts"].values())
    })
    
    colors = {
        "anger": "#FF4C4C",     
        "disgust": "#228B22",  
        "fear": "#9370DB",      
        "joy": "#FFD700",       
        "neutral": "#A9A9A9",   
        "sadness": "#1E90FF",   
        "surprise": "#FFA500"   
    }
    
    # pie chart with Plotly Express 
    fig = px.pie(
        df,
        values="Count",
        names="Emotion",
        title="Emotion Distribution",
        hover_data=["Count"],
        labels={"Count": "Count"},
        color="Emotion",
        color_discrete_map=colors
    )
    fig.update_traces(textposition="inside", textinfo="percent+label",hovertemplate="%{label}: %{percent} (%{value})")
    fig.update_layout(title={'x': 0, 'y': 0.95})
    st.plotly_chart(fig, use_container_width=True)
    st.write("Thank you for chatting!")
    
    # generate the emotional journey summary first
    try:
        emotion_summary = generate_emotion_story(
            st.session_state["conversation_memory"]["summaries"],
            st.session_state["emotion_counts"]
        )
    except Exception as e:
        st.error(f"❌ Error generating emotional story: {e}")
        emotion_summary = "Unable to generate emotional analysis at this time."

    # display the summary
    st.text_area("Emotional Journey Summary", value=emotion_summary, height=200)
    
        
    if st.button("New Chat"):
        st.session_state["chat_history"] = []
        st.session_state["full_context"] = ""
        st.session_state["last_processed_message"] = ""
        st.session_state["end_chat"] = False
        st.session_state["emotion_counts"] = {
            "anger": 0, "disgust": 0, "fear": 0,
            "joy": 0, "neutral": 0, "sadness": 0, "surprise": 0
        }
        st.session_state["sentiments"] = []
