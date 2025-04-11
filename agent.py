from model import detect_emotion, generate_response, generate_summary
import streamlit as st


conversation_memory = {
    "summaries": [],
    "previous_input": None,
    "previous_emotion_scores": None,
    "previous_bot_output": None,
    "is_first_prompt": True
}
def get_response_with_emotion_and_sentiment(user_input, sentiment, previous_context=""):
    if st.session_state["conversation_memory"] is None:
        st.session_state["conversation_memory"] = {
            "summaries": [],
            "previous_input": None,
            "previous_emotion_scores": None,
            "previous_bot_output": None,
            "is_first_prompt": True
        }
    
    top_emotion, emotion_scores = detect_emotion(user_input)

    if not st.session_state["conversation_memory"]["is_first_prompt"]:
        summary = generate_summary(
            st.session_state["conversation_memory"]["previous_input"],
            st.session_state["conversation_memory"]["previous_emotion_scores"],
            st.session_state["conversation_memory"]["previous_bot_output"]
        )
        st.session_state["conversation_memory"]["summaries"].append(summary)
    else:
        st.session_state["conversation_memory"]["is_first_prompt"] = False

    response = generate_response(
        user_input,
        sentiment,
        top_emotion,
        previous_context
    )

    st.session_state["conversation_memory"]["previous_input"] = user_input
    st.session_state["conversation_memory"]["previous_emotion_scores"] = emotion_scores
    st.session_state["conversation_memory"]["previous_bot_output"] = response

    return sentiment, top_emotion, response
