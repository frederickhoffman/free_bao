from typing import TypedDict, Annotated, List, Union, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import List, Dict, Any, Tuple
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
from free_bao.memory.mo_cer import MOCER

# --- Tools ---
@tool
def book_flight(destination: str, price_limit: int) -> str:
    """Books a flight to the destination if under price limit."""
    return f"Flight booked to {destination} for ${price_limit - 10}. Confirmation: #12345."

@tool
def search_hotels(location: str, date: str) -> str:
    """Searches for hotels in a location."""
    return f"Found 3 hotels in {location} for {date}: Hotel A ($100), Hotel B ($150)."

tools = [book_flight, search_hotels]

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    task: str
    context: str 
    steps: int

# --- Agent Class ---
class FreeBaoAgent:
    def __init__(self, memory: MOCER, model_name: str = "gpt-4o-mini"):
        self.memory = memory
        self.llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(tools)
        self.system_template = """You are a helpful and EFFICIENT assistant.
Your goal is to solve the user's task with the MINIMUM number of turns.
Avoid asking redundant questions. Infer what you can.
If you have enough information, ACT immediately using tools.

MEMORY OF EFFICIENT PAST SOLUTIONS:
{context}

Current Task: {task}
"""

    def retrieve_memory(self, state: AgentState):
        task = state["task"]
        context = self.memory.get_formatted_retrieval(task, k=1)
        return {"context": context}

    def reason(self, state: AgentState):
        messages = state["messages"]
        task = state["task"]
        context = state.get("context", "")
        
        # Ensure system message is first
        if not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(content=self.system_template.format(context=context, task=task))
            messages = [system_msg] + messages
        else:
            # Update system message content if needed
            messages[0] = SystemMessage(content=self.system_template.format(context=context, task=task))

        response = self.llm.invoke(messages)
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def should_continue(self, state: AgentState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # If no tool call, it means the agent is responding to the user
        # In a real app, this goes to the user. In simulation, we yield to user simulator.
        # But for the graph, we just end the turn.
        return "__end__"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve_memory)
        workflow.add_node("reason", self.reason)
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "reason")
        
        workflow.add_conditional_edges(
            "reason",
            self.should_continue,
        )
        workflow.add_edge("tools", "reason")
        
        return workflow.compile()
