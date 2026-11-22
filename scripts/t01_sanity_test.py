
import argparse
import json
import os
import re

SYSTEM = """
You are an agent solving household tasks in ALFWorld.

Respond using exactly this format:

Thought: Brief explanation of your next step
Action: One valid ALFWorld command

Valid actions include:
go to X, open X, close X, take X from Y,
put X in/on Y, use X, clean X with Y,
heat X with Y, cool X with Y, look, inventory.

Do not call tools or functions.
Do not return JSON.
"""


def parse_reply(text):
    """Extract thought and action from the model's response."""

    thought_match = re.search(r"Thought:\s*(.*)", text, re.I)
    action_match = re.search(r"Action:\s*(.*)", text, re.I)

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip() if action_match else "look"

    return thought, action

def run_sanity(step_fn, llm, obs, goal, max_steps=5):

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{obs}\nGoal: {goal}"}
    ]

    steps = []

    for t in range(1, max_steps + 1):

        reply = llm(messages)

        thought, action = parse_reply(reply)

        obs, done = step_fn(action)

        steps.append({
            "t": t,
            "thought": thought,
            "action": action,
            "observation": obs
        })

        messages.append({
            "role": "assistant",
            "content": reply
        })

        messages.append({
            "role": "user",
            "content": f"Observation: {obs}\nContinue solving the goal."
        })

        if done:
            break

    return {
        "goal": goal,
        "steps": steps
    }


def make_alfworld(config_path):

    import yaml
    from alfworld.agents.environment import get_environment

    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    env = get_environment(
        cfg["env"]["type"]
    )(cfg, train_eval="train").init_env(batch_size=1)

    obs, _ = env.reset()

    text = obs[0]

    goal = text.split("Your task is to:")[-1].strip()

    def step_fn(action):

        observation, _, done, _ = env.step([action])

        return observation[0], bool(done[0])

    return step_fn, text, goal


def make_groq_llm(model):

    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Run the API key cell first."
        )

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

    def llm(messages):

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

        message = response.choices[0].message

        if not message.content:
            raise ValueError(
                "The model returned an empty response."
            )

        return message.content

    return llm


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", default="t01_sample_output.json")
    parser.add_argument("--config", default="base_config.yaml")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")

    args = parser.parse_args()

    print("Initializing ALFWorld...")

    step_fn, obs, goal = make_alfworld(args.config)

    print("Connecting to Groq...")

    llm = make_groq_llm(args.model)

    print("Running agent...\n")

    result = run_sanity(
        step_fn,
        llm,
        obs,
        goal,
        args.steps
    )

    with open(args.out, "w") as file:
        json.dump(result, file, indent=2)

    print("\nGoal:", goal)

    for step in result["steps"]:

        print(
            f"{step['t']}. {step['action']} "
            f"-> {step['observation'][:70]}"
        )

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
