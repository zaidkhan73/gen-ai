from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("HuggingFace API Key:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", # this model cannot give structured output
    task="text generation",
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(
    llm=llm
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    input_variables=["input_text"],
    template="""write a detailed report on the following topic:
    {input_text}
    """
)

# 2nd prmpt -> detailed summary
template2 = PromptTemplate(
    input_variables=["input_text"],
    template="""write a summary on the following topic:
    {input_text}
    """
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({
    "input_text": "The impact of climate change on agriculture."
})
print(result)

# this code is about string output parser, which is used to parse the output of the model into a string format. 