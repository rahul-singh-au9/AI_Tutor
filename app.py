# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import gradio as gr
from huggingface_hub import snapshot_download

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata,QueryEngineTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()


# This section imports the necessary libraries and modules. Key components include:

# chromadb for vector storage.
# llama_index components for building the AI agent and query engine.
# gradio for creating the UI.
# We’re loading the environment variables from the .env file using load_dotenv().

# Next, we will prepare the prompts used by the Gradio app. The PROMPT_SYSTEM_MESSAGE prompt will be used by an agent, which will have to decide whether to use a tool named “AI_Information_related_resources” to retrieve useful AI information from a local knowledge base. The TEXT_QA_TEMPLATE prompt is used to write the final answer, leveraging the documents retrieved from the knowledge base.

PROMPT_SYSTEM_MESSAGE = """You are an AI teacher, answering questions from students of an applied AI course on Large Language Models (LLMs or llm) and Retrieval Augmented Generation (RAG) for LLMs. 
Topics covered include training models, fine-tuning models, giving memory to LLMs, prompting tips, hallucinations and bias, vector databases, transformer architectures, embeddings, RAG frameworks such as 
Langchain and LlamaIndex, making LLMs interact with tools, AI agents, reinforcement learning with human feedback (RLHF). Questions should be understood in this context. Your answers are aimed to teach 
students, so they should be complete, clear, and easy to understand. Use the available tools to gather insights pertinent to the field of AI.
To find relevant information for answering student questions, always use the "AI_Information_related_resources" tool.

Only some information returned by the tool might be relevant to the question, so ignore the irrelevant part and answer the question with what you have. Your responses are exclusively based on the output provided 
by the tools. Refrain from incorporating information not directly obtained from the tool's responses.
If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Provide 
comprehensive answers, ideally structured in multiple paragraphs, drawing from the tool's variety of relevant details. The depth and breadth of your responses should align with the scope and specificity of the information retrieved. 
Should the tool response lack information on the queried topic, politely inform the user that the question transcends the bounds of your current knowledge base, citing the absence of relevant content in the tool's documentation. 
At the end of your answers, always invite the students to ask deeper questions about the topic if they have any.
Do not refer to the documentation directly, but use the information provided within it to answer questions. If code is provided in the information, share it with the students. It's important to provide complete code blocks so 
they can execute the code when they copy and paste them. Make sure to format your answers in Markdown format, including code blocks and snippets.
"""

TEXT_QA_TEMPLATE = """
You must answer only related to AI, ML, Deep Learning and related concepts queries.
Always leverage the retrieved documents to answer the questions, don't answer them on your own.
If the query is not relevant to AI, say that you don't know the answer.
"""


# First, the code downloads the knowledge base locally (if it can’t find it already), then sets the models to use by the app (“gpt-4o-mini” as the LLM, “text-embedding-3-small” as the embedding model), and finally, it launches the Gradio UI.

# Let’s inspect the code of the download_knowledge_base_if_not_exists function.


def download_knowledge_base_if_not_exists():
    """Download the knowledge base from the Hugging Face Hub if it doesn't exist locally"""
    if not os.path.exists("data/ai_tutor_knowledge"):
        os.makedirs("data/ai_tutor_knowledge")

        logging.warning(
            f"Vector database does not exist at 'data/', downloading from Hugging Face Hub..."
        )
        snapshot_download(
            repo_id="jaiganesan/ai_tutor_knowledge_vector_Store",
            local_dir="data/ai_tutor_knowledge",
            repo_type="dataset",
        )
        logging.info(f"Downloaded vector database to 'data/ai_tutor_knowledge'")

# The above code creates the “data/ai_tutor_knowledge” folder if it doesn’t exist yet, and then downloads the content of the jaiganesan/ai_tutor_knowledge_vector_Store Hugging Face dataset into it. It will download a Chroma vector store with the 500 blog dataset and their embeddings.


# The Gradio UI
# Let’s see the code of the launch_ui function now.


# It starts by creating a Gradio Blocks context. Blocks are Gradio’s low-level API that allows you to create more custom web applications and demos than the usual approach using Gradio’s Interfaces. 

# This code defines a function called launch_ui() that creates and launches a simple AI tutor chatbot interface using the Gradio library. Here’s a step-by-step explanation of the main parts:

# Creating the interface with gr.Blocks:
# gr.Blocks are components that arrange elements in blocks, providing a container for the interface.
# Several parameters are specified, including fill_height=True (to adjust the height of elements) and title= “AI Tutor 🤖” which sets the title of the browser tab that contains the app.
# The as demo: part assigns this Blocks component to demo, so everything inside will be part of the demo app.
# Setting up memory with gr.State:
# gr.State holds the chatbot’s memory state. This is initialized with ChatSummaryMemoryBuffer (a structure for summarizing and storing chat history).
# The memory buffer has a token_limit=120000, setting a maximum token count to limit the amount of text the AI retains.
# Creating the chatbot interface with gr.Chatbot:
# gr.Chatbot defines the actual chatbot’s appearance and behavior.
# scale=1 sets the visual scale. The placeholder provides an introductory message, styling it as a bold header describing the bot as an “AI Tutor.”
# show_copy_button=True enables a copy button for easy sharing of responses.
# Defining the chat logic with gr.ChatInterface:
# gr.ChatInterface sets up the function for generating responses, generate_completion, as the chatbot’s main logic.
# It connects the chatbot instance and provides additional_inputs (here, memory_state) so the bot remembers the conversation context.
# Setting up and launching the app:
# demo.queue(default_concurrency_limit=64) limits the number of users who can interact with the bot simultaneously.
# demo.launch(debug=True, share=False) starts the app, with debug=True to display errors for troubleshooting and share=False to keep the app private by default.
# With Gradio, there are many UI components that you can easily use for your app. Visit this documentation page to learn more.

def launch_ui():
    with gr.Blocks(
        fill_height=True,
        title="AI Tutor 🤖",
        analytics_enabled=True,
    ) as demo:

        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        chatbot = gr.Chatbot(
            scale=1,
            placeholder="<strong>AI Tutor 🤖: A Question-Answering Bot for anything AI-related</strong><br>",
            show_label=False,
            show_copy_button=True,
        )

        gr.ChatInterface(
            fn=generate_completion,
            chatbot=chatbot,
            additional_inputs=[memory_state],
        )

        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, share=False) # Set share=True to share the app online


# Generating Responses
# Now, let’s inspect the implementation of the generate_completion function.

def generate_completion(query, history, memory):
    logging.info(f"User query: {query}")

    # Manage memory
    chat_list = memory.get()
    if len(chat_list) != 0:
        user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
        if len(user_index) > len(history):
            user_index_to_remove = user_index[len(history)]
            chat_list = chat_list[:user_index_to_remove]
            memory.set(chat_list)
    logging.info(f"chat_history: {len(memory.get())} {memory.get()}")
    logging.info(f"gradio_history: {len(history)} {history}")

    # Create agent
    tools = get_tools(db_collection="ai_tutor_knowledge")
    agent = OpenAIAgent.from_tools(
        llm=Settings.llm,
        memory=memory,
        tools=tools,
        system_prompt=PROMPT_SYSTEM_MESSAGE,
    )

    # Generate answer
    completion = agent.stream_chat(query)
    answer_str = ""
    for token in completion.response_gen:
        answer_str += token
        yield answer_str


# This code defines the function generate_completion, which processes user queries, manages the chatbot’s memory, creates an AI agent, and generates a response to the query. Here’s a breakdown of each part:

# 1. Chat memory management:
# The function retrieves the current memory of past chats using memory.get(), which returns a list of chat messages.
# The rest of the code keeps the chat memory in sync with the current chat in the UI.
# After updating memory, the function logs the current memory length and chat history length to provide context on memory and interaction states.

# 2. Creating the AI Agent:
# The function sets up an AI agent that will respond to the user. It creates a set of tools by calling get_tools, which connects to a database of knowledge (db_collection= “ai_tutor_knowledge”) relevant to the AI Tutor. We’ll see more about this tool list later. It contains only one tool which connects the agent to the knowledge base.
# It then initializes an OpenAIAgent with several components:
# llm specifies the language model to use.
# memory is the managed chat memory.
# tools are resources the agent can use to answer questions.
# system_prompt provides system-level instructions for the agent’s behavior (which we saw earlier in this lesson).

# 3. Generating a response:
# The agent.stream_chat(query) method is used to generate and stream a response based on the user’s query.
# The response is built token by token in a generator loop, which yields the progressively constructed answer back to the caller in real time. The tokens are concatenated to form answer_str, and each new token is yielded as it arrives, allowing the chatbot to “stream” the answer back to the user.


# Tools
# Last, let’s see the implementation of the get_tools function.


def get_tools(db_collection="ai_tutor_knowledge"):
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=15,
        embed_model=Settings.embed_model,
        use_async=True,
    )
    tools = [
        RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="AI_Information_related_resources",
                description="Useful for info related to artificial intelligence, ML, deep learning. It gathers the info from local data.",
            ),
        )
    ]
    return tools

# It loads the Chroma vector database that we downloaded first thing in the code, then creates a retriever out of it and leverages it to create a RetrieverTool object. Notice that the tool's name, “AI_Information_related_resources,” is the same as mentioned in the system prompt passed to the agent.

# Launch the App
# We are now ready to launch and test the app! Run python app.py from your terminal. After the knowledge base has been downloaded, it should launch a server accessible at the local URL http://127.0.0.1:7860. 


# Main Function
if __name__ == "__main__":
    # Download the knowledge base if it doesn't exist
    download_knowledge_base_if_not_exists()

    # Set up llm and embedding model
    Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # launch the UI
    launch_ui()