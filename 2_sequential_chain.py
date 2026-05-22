from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["LANGCHAIN_PROJECT"] = 'Sequential LLM App'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model_1 = ChatGoogleGenerativeAI(
    model=os.getenv("SELECTED_MODEL"),
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)


model_2 = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parser = StrOutputParser()

chain = prompt1 | model_1 | parser | prompt2 | model_2 | parser

config = {
    'run_name': 'Sequential LLM App Run',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {
        'author': 'Raj',
        'description': 'An app that generates a report on a given topic and then summarizes it into 5 pointers.',
        'model': 'gemini-2.0-flash',
        'model_temperature': 0.7,
        'parser': 'StrOutputParser'
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
