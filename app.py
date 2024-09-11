import streamlit as st
import os
from groq import Groq
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import tempfile

def extract_text_from_pdf_directory(pdf_dir):
    """Extract text from a directory containing PDF files using PyPDFDirectoryLoader."""
    try:
        loader = PyPDFDirectoryLoader(pdf_dir)
        documents = loader.load()
        text = "\n\n".join([doc.page_content for doc in documents])
    except Exception as e:
        st.error(f"Error extracting text from PDF directory: {e}")
        return ""
    return text

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        st.error("API key not found. Please set the 'GROQ_API_KEY' environment variable.")
        return

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Chat with Groq!")
    st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Initialize a variable to hold the extracted PDF text
    extracted_pdf_text = ""

    # File uploader for PDF
    pdf_file = st.file_uploader("Upload a PDF file:", type="pdf")
    if pdf_file:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save the uploaded file to the temporary directory
            file_path = os.path.join(tmpdirname, pdf_file.name)
            with open(file_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Extract text from the PDF in the temporary directory
            extracted_pdf_text = extract_text_from_pdf_directory(tmpdirname)
            st.text_area("Extracted Text from PDF:", extracted_pdf_text, height=300)

    user_question = st.text_input("Ask a question:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # If the PDF has been uploaded and text is extracted
    if extracted_pdf_text:
        # Modify the system prompt to include the extracted PDF text
        system_prompt = f"Here is the content of the PDF: {extracted_pdf_text}\n{system_prompt}"

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='llama-3.1-70b-versatile'  # Fixed model as requested
    )

    # If the user has asked a question,
    if user_question:

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt  # Include the system prompt with the PDF content
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=prompt,  # The constructed prompt template.
            verbose=True,   # Enables verbose output, which can be useful for debugging.
            memory=memory,  # The conversational memory object that stores and manages the conversation history.
        )

        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

        # Display conversation history with better formatting
        st.write("### Conversation History:")
        for i, msg in enumerate(st.session_state.chat_history):
            # Format human message in blue and AI message in gray for better visualization
            st.markdown(f"<div style='padding: 10px; background-color: #e6f7ff; border-radius: 10px; margin-bottom: 10px;'><b>You:</b> {msg['human']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 10px; background-color: #f2f2f2; border-radius: 10px;'><b>AI:</b> {msg['AI']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
