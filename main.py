from interface import *

tset_mode =[
    "Normal",           # 0, the full PReP Agent
    "NoReflection",     # 1, PReP Agent w/o Reflection
    "NoPlanning",       # 2, PReP Agent w/o Planning
    "Plain",            # 3, PReP Agent w/o Reflection & Planning
    "CoT",              # 4, Chain-of_Thought Agent
    "Cap",              # 5, Cap Agent
    "Progprompt",       # 6, Progprompt Agent
    "InnerMonologue",   # 7, InnerMonologue Agent
    "DEPS",             # 8, DEPS Agent
    "Oracle",           # 9, PReP Agent with Oracle Perception
    "WithoutFinetune",  # 10, PReP Agent with un-finetuned Percetion
]


if __name__ == "__main__":

    llm_model = {
        'name': "gpt-4o-mini",
        'api_base': None
    }

    city = "newyork"

    mode = 0

    test_tag = tset_mode[mode]    # or other tag you want

    agent_test(city, llm_model, test_tag, mode)