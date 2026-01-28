import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ---------- Setup ----------
st.title("Chat-App")
load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Session State -
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content":( 
                "You are a chat assistant.\n"
                "You MUST NOT reveal chain-of-thought.\n"
                "Do NOT output <think> blocks.\n"
                "If you generate reasoning internally, do NOT print it.\n"
                "ONLY output the final answer to the user.")
        }
    ]

#  Display Chat History 
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.text_area("Enter your prompt")

if prompt:
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    #  Assistant Streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1:novita",
            messages=st.session_state.messages,
            stream=True,
        )

        

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_response += delta.content
                message_placeholder.markdown(strip_think(full_response) + "▌")


        final_answer = strip_think(full_response)
        message_placeholder.markdown(final_answer)


    # Save assistant response
    st.session_state.messages.append({
    "role": "assistant",
    "content": final_answer
        })
