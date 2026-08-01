from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os


from langchain_core.prompts import PromptTemplate
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

prompt1 = template1.invoke({"input_text": "The impact of climate change on agriculture."})
prompt2 = template2.invoke({"input_text": "The impact of climate change on agriculture."})

result1 = model.invoke(prompt1)
result2 = model.invoke(prompt2)

print("Detailed Report:\n", result1.content)
print("\nSummary:\n", result2.content)