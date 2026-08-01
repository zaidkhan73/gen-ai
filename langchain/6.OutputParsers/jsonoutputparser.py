from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import os


from langchain_core.prompts import PromptTemplate
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", # this model cannot give structured output
    task="text generation",
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(
    llm=llm
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instructions}',
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format()

print("Prompt:\n", prompt)

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print("Final Result:\n", final_result)
print("type of final result:", type(final_result))

# with json output parser, the result is in form of json format
# but you can not enforce any schema for the json result, that will be decided by the model itself.
# if you want to enforce a schema for the json result, you can use structured output parser instead of json output parser.