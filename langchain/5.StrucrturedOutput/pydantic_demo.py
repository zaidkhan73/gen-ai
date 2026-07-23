# pydantic: data validation and data parsing library for python.
# ensures that data is structured correct and type safe.

from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa : float = Field(..., gt=0, lt=10, description="CGPA must be between 0 and 10", default=5.8)   

new_student = student(name="zaid", email="zaid@example.com", cgpa=8.5)
print(new_student.name)  # Output: zaid
print(new_student.email)  # Output: zaid@example.com
print(new_student.cgpa)  # Output: 8.5