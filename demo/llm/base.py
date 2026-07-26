"""
This module contains the BaseLLM class, which is used as a base class for all LLMs.
"""
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    A base class for all LLMs.
    """
    @abstractmethod
    def generate_response(self, user: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        """
