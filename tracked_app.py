
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Step 1: Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError("LANGSMITH_API_KEY not found in .env")

# Step 2: Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "LLM-Question-Answer-Project"

print("API keys loaded!")
print("LangSmith tracing enabled!")

# Step 3: Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Step 4: Load questions from a text file
with open("questions.txt", "r", encoding="utf-8") as file:
    questions = [
        line.strip()
        for line in file
        if line.strip()
    ]

# Step 5: Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

# Step 6: Create a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.
Answer the following question clearly and simply.

Question:
{question}

Answer:
"""
)

# Step 7: Connect prompt and LLM
chain = prompt | llm

# Step 8: Process each question
for i, question in enumerate(questions, start=1):

    print(f"\n--- Question {i} ---")
    print(question)

    # Split long questions into chunks if needed
    chunks = text_splitter.split_text(question)

    for j, chunk in enumerate(chunks, start=1):
        response = chain.invoke(
            {"question": chunk},
            config={
                "run_name": f"Answer-Question-{i}-Chunk-{j}",
                "tags": ["llm", "prompt-template", "qa"],
                "metadata": {
                    "question_number": i,
                    "chunk_number": j
                }
            }
        )

        print("\nLLM Answer:")
        print(response.content)

print("\nAll questions processed!")
print("Check LangSmith to view your traces.")
