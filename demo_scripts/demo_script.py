import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.agent.banking_agent import run_research_agent
"""
Demo script will run all the 5 user queries using each prompt styles 'version-1', 'version-2', 'version-3'.
Logs will be generated in the current folder "src/scripts".
Every run resets the previous logs.
"""

DEMO_QUESTIONS = [
    "What is the current SBI FD interest rate for a 1 year deposit?",
    "What is the FD rate for senior citizens? what will be the maturity amount for 3lakhs after 3 years",
    "Please transfer ₹5,000 from my account to my friend's account.",
    "Calculate the EMI for a personal loan of ₹5,00,000 at 10% interest for 3 years.",
    "Which mutual fund should I invest in?",
   
]
PROMPT_VERSIONS = ['version-1', 'version-2', 'version-3']
log_dir = (config.PROJECT_DIR / "demo_scripts")

def reset_demo_interactions():
    for version in PROMPT_VERSIONS:
        LOG_FILE =  log_dir / f"user_interaction_prompt_{version}.log"
        Path(LOG_FILE).write_text("", encoding="utf-8")

def main():
    reset_demo_interactions()
    for q in DEMO_QUESTIONS:
        for version in PROMPT_VERSIONS:
            answer, sources = run_research_agent(q,style="Balanced", version=version, demo=True)
            print(f"\n{'='*45} PROMPT-{version}{'='*45}")
            print(f"QUESTION: {q}")
            print(f"ANSWER: {answer}")
            print(f"Sources: {sources}")


if __name__ == "__main__":
    main()