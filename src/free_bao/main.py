import argparse
import os
from free_bao.utils import load_keys_from_bashrc
from free_bao.memory.mo_cer import MOCER
from free_bao.simulation.benchmark import BenchmarkRunner
from free_bao.agent.react_agent import FreeBaoAgent

def main():
    load_keys_from_bashrc()
    
    parser = argparse.ArgumentParser(description="FREE-BAO: Contextual Experience Replay for Agents")
    parser.add_argument("--mode", choices=["ui", "benchmark"], default="benchmark", help="Mode to run")
    parser.add_argument("--benchmark-mode", choices=["warmup", "eval"], default="eval", help="benchmark phase")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--alpha", type=float, default=0.1, help="Pareto weight for efficiency (alpha)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset file (csv/json)")
    
    args = parser.parse_args()
    
    memory = MOCER(alpha=args.alpha)
    
    if args.mode == "benchmark":
        runner = BenchmarkRunner(memory, dataset_path=args.dataset)
        runner.run_benchmark(num_episodes=args.episodes, mode=args.benchmark_mode)
        
    elif args.mode == "ui":
        print("Starting LangGraph UI mode (Simulated CLI for now)...")
        agent = FreeBaoAgent(memory)
        app = agent.build_graph()
        
        print("Ask me to book a flight or find a hotel!")
        while True:
            try:
                user_input = input("User: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                # In a real UI, we'd maintain state across the loop
                # Here we just show a single-turn invocation for demo
                state = {"messages": [("user", user_input)], "task": user_input, "steps": 0}
                for event in app.stream(state):
                    for key, value in event.items():
                        if "messages" in value:
                            print(f"Agent ({key}): {value['messages'][-1].content}")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
