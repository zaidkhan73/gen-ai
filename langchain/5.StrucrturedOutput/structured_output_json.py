import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------
load_dotenv()

# -------------------------------------------------------
# Initialize Gemini Model
# -------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# -------------------------------------------------------
# Load JSON Schema
# -------------------------------------------------------
with open("json_schema.json", "r") as f:
    student_schema = json.load(f)

# -------------------------------------------------------
# Create Structured Model using JSON Schema
# -------------------------------------------------------
structured_model = model.with_structured_output(student_schema)

# -------------------------------------------------------
# User Input
# -------------------------------------------------------
student_info = """
My name is Zaid Khan.
I am 20 years old.
My email is zaidkhan@gmail.com.
My CGPA is 8.72.
"""

# -------------------------------------------------------
# Invoke Model
# -------------------------------------------------------
result = structured_model.invoke(student_info)

# -------------------------------------------------------
# Print Output
# -------------------------------------------------------
print(result)