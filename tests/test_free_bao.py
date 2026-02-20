import pytest
from unittest.mock import MagicMock, patch
from memory.memory import FreeBaoMemory, Episode
from agent.react_agent import FreeBaoAgent
from simulation.user_simulator import UserSimulator
from langchain_core.messages import AIMessage, HumanMessage

@pytest.fixture
def mock_memory():
    # Mocking ChromaDB client would be complex, so we mock MOCER methods directly if possible.
    # But MOCER uses persistent client. Let's start with a fresh persistent dir for tests or mock it.
    with patch("memory.memory.chromadb.Client") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock retrieval
        memory = FreeBaoMemory(persist_directory="./test_db")
        memory.collection = mock_collection
        # Mock embedding model
        memory.model = MagicMock()
        memory.model.encode.return_value = [0.1] * 384
        
        return memory

def test_mo_cer_add_episode(mock_memory):
    episode = Episode(
        task_description="Test Task",
        trajectory="User: hi\nAgent: hello",
        success=True,
        turns=2,
        metadata={}
    )
    mock_memory.add_episode(episode)
    mock_memory.collection.add.assert_called_once()

def test_mo_cer_retrieve_pareto(mock_memory):
    # Mock query results
    mock_memory.collection.query.return_value = {
        "ids": [["1", "2"]],
        "documents": [["traj1", "traj2"]],
        "metadatas": [[{"turns": 5, "task": "t1"}, {"turns": 2, "task": "t2"}]],
        "distances": [[0.5, 0.6]] # t1 is closer but t2 is more efficient
    }
    
    # retrieve_pareto_efficient calculates score = distance + (turns * 0.1)
    # t1 score = 0.5 + 0.5 = 1.0
    # t2 score = 0.6 + 0.2 = 0.8  <- Winner
    
    results = mock_memory.retrieve_pareto_efficient("query context")
    assert len(results) == 2
    assert results[0]["turns"] == 2 # Should be first
    assert results[1]["turns"] == 5

@patch("agent.react_agent.ChatOpenAI")
def test_agent_graph_build(mock_chat, mock_memory):
    agent = FreeBaoAgent(mock_memory)
    app = agent.build_graph()
    assert app is not None

@patch("simulation.user_simulator.ChatOpenAI")
def test_user_simulator(mock_chat):
    sim = UserSimulator()
    mock_chat.return_value.invoke.return_value = AIMessage(content="I want a flight")
    
    response = sim.step("Hello", "Goal", [])
    assert response == "I want a flight"
