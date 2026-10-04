import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# ==========================================
# Create Agent only ONCE
# ==========================================

@st.cache_resource
def create_my_agent():

    llm = init_chat_model(
        model="groq:openai/gpt-oss-120b"
    )

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[],
        checkpointer=memory,
        system_prompt="You are a helpful assistant."
    )

    return agent


agent = create_my_agent()


# ==========================================
# Thread ID
# ==========================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "conversation-1"


config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ==========================================
# UI
# ==========================================

st.title("Q&A Bot with Memory 🧠")

question = st.text_input("Your question")


if st.button("Ask") and question:

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=config
    )

    print("\nYou:", question)
    print("Bot:", response["messages"][-1].content)

    st.write(response["messages"][-1].content)


    # DEBUG: দেখুন memory-তে কী আছে
    st.write("### Messages currently in State")

    state = agent.get_state(config)

    for message in state.values["messages"]:
        st.write(
            type(message).__name__,
            ":",
            message.content
        )