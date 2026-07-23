from typing import TypedDict, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#schema
class review(TypedDict):
    summary: Annotated[str, "A concise summary of the review"] # Annotated is used to provide additional metadata or context about the type, in this case, a description of what the string represents.
    sentiment: Annotated[str, "The sentiment of the review"]

structured_model = model.with_structured_output(review)

result = structured_model.invoke(
    "'I love using LangChain for building AI applications!'"
) 

print("summary:", result["summary"])
print("sentiment:", result["sentiment"])


# typeddict is a way to define a dictionary with specific keys and value types in Python. It allows you to create a structured representation of data, where each key has an associated type. In this example, the review TypedDict defines two keys: summary and sentiment, both of which are strings. The Annotated type is used to provide additional metadata or context about the type, in this case, a description of what the string represents.

# annotated is a way to add metadata or context to a type in Python. It allows you to provide additional information about the type, such as a description or constraints. In this example, the Annotated type is used to provide descriptions for the summary and sentiment keys in the review TypedDict.

# literal is a way to define a type that can only take on specific values in Python. It allows you to restrict the possible values of a variable to a predefined set. In this example, the sentiment key in the review TypedDict could be defined as a Literal type with specific values like "positive", "negative", or "neutral" to indicate the sentiment of the review.