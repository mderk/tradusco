"""
Utility functions and classes for the translation project.
"""

from pydantic import BaseModel


class Config(BaseModel):
    """Project configuration model"""

    name: str
    sourceFile: str
    languages: list[str]
    baseLanguage: str
    keyColumn: str
