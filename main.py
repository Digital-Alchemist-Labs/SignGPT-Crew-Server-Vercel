from crew import SginGPTCrew
import os
import json
from dotenv import load_dotenv

load_dotenv()


def main():

    with open("./data/english_words.json", "r") as f:
        asl_dataset = json.load(f)

    asl_dataset = [asl_dataset[word].upper() for word in asl_dataset]

    words = ["YOU", "NAME", "WHAT", "YOU", "DO", "WHAT"]

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Set it in your environment or a .env file at repo root."
        )

    result = SginGPTCrew().sgin_gpt_crew().kickoff(
        inputs={'words': words, 'ASL_dataset': asl_dataset})
    # Print final output
    print("Crew Result:", result)

    # Print all step outputs at the end
    print("\n==== All Step Outputs ====")
    for idx, task_output in enumerate(result.tasks_output, start=1):
        step_name = task_output.description or f"Step {idx}"
        agent_name = task_output.agent or "Unknown Agent"
        print(f"[{idx}] {step_name} — Agent: {agent_name}")
        print(f"Output: {task_output.raw}\n")


if __name__ == "__main__":
    main()
