import pandas as pd

from langchain_aws import ChatBedrock
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents.agent_types import AgentType


from dataclasses import dataclass
from typing import Literal
import streamlit as st

from langchain import OpenAI
# from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components
# from langchain_core.runnables.history import RunnableWithMessageHistory

@dataclass
class Message:
    
    """Class for keeping track of a chat message."""
    
    origin: Literal["human", "ai"]
    message: str


def load_css():
    
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def initialize_session_state():
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    
    
    if "conversation" not in st.session_state:
        
        # llm = OpenAI(
        #     temperature=0,
        #     openai_api_key=st.secrets["openai_api_key"],
        #     model_name="text-davinci-003"
        # )
        llm = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs=dict(temperature=0),
            aws_access_key_id=st.secrets['aws_access_key_id'],
            aws_secret_access_key=st.secrets['aws_secret_access_key'],
            region_name='us-east-1'
        )

        # RunnableWithMessageHistory()
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=ConversationSummaryMemory(llm=llm),
        )


def on_click_callback():
    
    with get_openai_callback() as cb:
        
        human_prompt = st.session_state.human_prompt
        llm_response = st.session_state.conversation.invoke(
            human_prompt
        )
        
        st.session_state.history.append(
            Message("human", human_prompt)
        )
        
        st.session_state.history.append(
            Message("ai", llm_response.get('response'))
        )
        
        st.session_state.token_count += cb.total_tokens


load_css()
initialize_session_state()

st.title("Inmobiliaria Chatbot 🤖")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")
credit_card_placeholder = st.empty()

with chat_placeholder:
    
    for chat in st.session_state.history:
        div = f"""
            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                <img class="chat-icon" src="app/static/{'asistente-de-ai.png' if chat.origin == 'ai' else 'usuario.png'}" width=32 height=32>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)

    for _ in range(3):
        st.markdown("")


with prompt_placeholder:
    
    st.markdown("**Chat**")
    
    cols = st.columns((6, 1))
    
    
    cols[0].text_input(
        "Chat",
        value="Hola bot",
        label_visibility="collapsed",
        key="human_prompt",
    )
    
    
    cols[1].form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,
    )



credit_card_placeholder.caption(
    f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain conversation: 
    {st.session_state.conversation.memory.buffer}
    """
)

components.html("""
<script>
const streamlitDoc = window.parent.document;

const buttons = Array.from(
    streamlitDoc.querySelectorAll('.stButton > button')
);
const submitButton = buttons.find(
    el => el.innerText === 'Submit'
);

streamlitDoc.addEventListener('keydown', function(e) {
    switch (e.key) {
        case 'Enter':
            submitButton.click();
            break;
    }
});
</script>
""",
    height=0,
    width=0,
)

#!


# PATH_PROCESSED = './data/processed/'
# data = pd.read_excel(PATH_PROCESSED + 'inmoDataProcessed.xlsx')


# llm = ChatBedrock(
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             model_kwargs=dict(temperature=0),
#             aws_access_key_id=st.secrets['aws_access_key_id'],
#             aws_secret_access_key=st.secrets['aws_secret_access_key'],
#             region_name='us-east-1'
#         )

# llm.invoke(input="Cuál es la capital de Ecuador?")

# messages = [
#     (
#         "system",
#         "Eres un asesor inmobiliario que responde en español",
#     ),
#     ("human", "Cuál es la capital de Ecuador?"),
# ]
# ai_msg = llm.invoke(messages)
# ai_msg

# agent = create_pandas_dataframe_agent(
#     llm=llm,
#     df=data,
#     verbose=False,
#     allow_dangerous_code=True,
#     agent_executor_kwargs={'handle_parsing_errors': True}
# )


# messages = [
#     (
#         "system",
#         "Eres un asesor inmobiliario que habla en español",
#     ),
#     ("human", "Cuántas filas y columnas existen?"),
# ]
# ai_msg = agent.invoke(messages)
# ai_msg.get('output')


# messages = [
#     (
#         "system",
#         "Eres un asesor inmobiliario que responde en español",
#     ),
#     ("human", "Puedes decirme cuántos TIPOS existen y cuáles son?"),
# ]
# ai_msg = agent.invoke(messages)
# print(ai_msg.get('output'))
