from dotenv import load_dotenv
from typing import Optional, List, Literal

from pydantic import BaseModel, EmailStr, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------------------------------------
# Load Environment Variables (.env)
# -------------------------------------------------------
load_dotenv()

# -------------------------------------------------------
# Initialize Gemini Model
# -------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# -------------------------------------------------------
# Define Pydantic Schema
# This tells the LLM exactly what output format we want.
# -------------------------------------------------------

class ProductReview(BaseModel):

    # Product Name
    product_name: str = Field(
        description="Name of the reviewed product"
    )

    # Short Summary
    summary: str = Field(
        description="Short summary of the review"
    )

    # Rating (1-5)
    rating: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Overall rating between 1 and 5"
    )

    # Positive Points
    pros: List[str] = Field(
        description="List of positive features"
    )

    # Negative Points
    cons: List[str] = Field(
        description="List of negative features"
    )

    # Recommended?
    recommended: bool = Field(
        description="Whether the reviewer recommends the product"
    )

    # Sentiment
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Overall sentiment"
    )

    # Reviewer's Email (Optional)
    reviewer_email: Optional[EmailStr] = Field(
        default=None,
        description="Reviewer's email if available"
    )

# -------------------------------------------------------
# Tell LangChain to return output in ProductReview format
# -------------------------------------------------------

structured_model = model.with_structured_output(ProductReview)

# -------------------------------------------------------
# Input Review
# -------------------------------------------------------

review = """
I recently bought the Sony WH-1000XM5 headphones.

The sound quality is amazing and the noise cancellation is outstanding.
Battery backup easily lasts two days.

However, the price is a little high and the carrying case is bulky.

Overall I highly recommend these headphones.
"""

# -------------------------------------------------------
# Invoke LLM
# -------------------------------------------------------

result = structured_model.invoke(review)

# -------------------------------------------------------
# Print Structured Output
# -------------------------------------------------------

print("\n=========== Structured Output ===========\n")

print("Product Name :", result.product_name)
print("Summary      :", result.summary)
print("Rating       :", result.rating)
print("Pros         :", result.pros)
print("Cons         :", result.cons)
print("Recommended  :", result.recommended)
print("Sentiment    :", result.sentiment)
print("Email        :", result.reviewer_email)