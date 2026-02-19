from typing import Dict, List, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os

class UserSimulator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.system_prompt = """You are a user interacting with an AI assistant.
You have a specific GOAL that you want the assistant to help you with.
The assistant does not know your goal initially.
Answer the assistant's questions truthfully based on your goal.
Do not reveal the entire goal at once unless asked specifically.
Be impatient if the assistant asks redundant or irrelevant questions.
Your goal is: {goal}
"""

    def step(self, agent_last_message: str, goal: str, history: List[BaseMessage]) -> str:
        """Generates the user's response."""
        messages = [
            SystemMessage(content=self.system_prompt.format(goal=goal)),
            *history,
            AIMessage(content=agent_last_message)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
