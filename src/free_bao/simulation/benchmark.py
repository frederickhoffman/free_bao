import wandb
import pandas as pd
from tqdm import tqdm
from free_bao.agent.react_agent import FreeBaoAgent, AgentState
from free_bao.simulation.user_simulator import UserSimulator
from free_bao.memory.memory import FreeBaoMemory, Episode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List

# --- Synthetic Dataset ---
TASKS = [
    {"goal": "I want to fly to Paris for under $500", "task": "Help the user book a flight."},
    {"goal": "I need a hotel in Tokyo for March 1st, budget is flexible but prefer under $200", "task": "Find a hotel for the user."},
    {"goal": "Book a flight to NY, cheapest possible, I don't care about layovers", "task": "Book a flight."},
    {"goal": "I want a flight to London, business class, any price", "task": "Book a flight."},
    {"goal": "Find me a hotel in Berlin for next weekend", "task": "Find a hotel."},
]

class BenchmarkRunner:
    def __init__(self, memory: FreeBaoMemory, project_name: str = "free_bao_benchmark", dataset_path: str = None):
        self.memory = memory
        self.user_sim = UserSimulator()
        self.project_name = project_name
        self.dataset = self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str = None) -> List[dict]:
        if dataset_path:
            # Simple assumption: CSV with 'goal' and 'task' columns
            # Or JSON list of dicts
            if dataset_path.endswith(".csv"):
                 df = pd.read_csv(dataset_path)
                 return df.to_dict(orient="records")
            elif dataset_path.endswith(".json"):
                 import json
                 with open(dataset_path, "r") as f:
                     return json.load(f)
            else:
                 print(f"Unknown file extension for {dataset_path}, falling back to synthetic.")
                 return TASKS
        return TASKS

    def run_benchmark(self, num_episodes: int = 5, mode: str = "eval", warmup_episodes: int = 0):
        """
        Runs the benchmark.
        mode: 'warmup' (populate memory) or 'eval' (measure performance)
        warmup_episodes: Number of episodes to run as 'warmup' before starting 'eval'
        """
        # If eval mode and warmup_episodes requested, run warmup first
        if mode == "eval" and warmup_episodes > 0:
            print(f"--- Starting INTERNAL WARMUP ({warmup_episodes} episodes) ---")
            self._execute_phase(warmup_episodes, "warmup")
            print(f"--- INTERNAL WARMUP COMPLETE ---\n")

        # Run the main phase
        self._execute_phase(num_episodes, mode)

    def _execute_phase(self, num_episodes: int, mode: str):
        """Internal method to execute a specific benchmark phase."""
        run = wandb.init(project=self.project_name, job_type=mode, config={"alpha": self.memory.alpha}, reinit=True)
        columns = ["task", "success", "turns", "trajectory", "mode"]
        table = wandb.Table(columns=columns)
        
        results = []
        
        print(f"Starting {mode} phase with {num_episodes} episodes using {len(self.dataset)} tasks...")
        
        for i in range(num_episodes):
            dataset_item = self.dataset[i % len(self.dataset)]
            goal = dataset_item["goal"]
            task = dataset_item["task"]
            
            agent = FreeBaoAgent(self.memory)
            app = agent.build_graph()
            
            # Initial state
            state = {
                "messages": [HumanMessage(content=task)],
                "task": task,
                "steps": 0
            }
            
            history: List[BaseMessage] = []
            trajectory = ""
            success = False
            
            # Interaction Loop
            current_messages = [HumanMessage(content=task)]
            
            for step in range(10): # Max 10 turns
                # Agent acts
                result = app.invoke({"messages": current_messages, "task": task})
                
                output_messages = result["messages"]
                new_agent_messages = output_messages[len(current_messages):]
                current_messages = output_messages # Update our view of state
                
                if not new_agent_messages:
                    break
                    
                # Scan through new messages for ToolMessage indicating success
                for msg in new_agent_messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                         trajectory += f"Agent Tool Call: {msg.tool_calls[0]['name']}\n"
                    elif msg.type == "tool":
                        trajectory += f"Tool Output: {msg.content}\n"
                        if "Booked" in str(msg.content) or "Found" in str(msg.content):
                            success = True
                    elif msg.type == "ai":
                        trajectory += f"Agent: {msg.content}\n"
                        if msg.content:
                             history.append(msg)

                if success:
                    break
                
                # Get the final response text to send to user
                last_response = ""
                for msg in reversed(current_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        last_response = msg.content
                        break
                
                # User Sim responds
                user_response = self.user_sim.step(last_response, goal, history)
                trajectory += f"User: {user_response}\n"
                
                user_msg = HumanMessage(content=user_response)
                history.append(user_msg)
                current_messages.append(user_msg)
            
            turns = step + 1
            
            # Log to WandB
            table.add_data(task, success, turns, trajectory, mode)
            results.append({"success": success, "turns": turns})
            
            # If warmup and successful, add to memory
            if mode == "warmup" and success:
                self.memory.add_episode(Episode(
                    task_description=task,
                    trajectory=trajectory,
                    success=success,
                    turns=turns,
                    metadata={"goal": goal}
                ))
                
        wandb.log({"results_table": table})
        
        if mode == "eval":
            avg_turns = sum(r["turns"] for r in results) / len(results)
            success_rate = sum(1 for r in results if r["success"]) / len(results)
            wandb.log({"avg_turns": avg_turns, "success_rate": success_rate})
            print(f"Eval Results - Avg Turns: {avg_turns}, Success Rate: {success_rate}")
            
        run.finish()

