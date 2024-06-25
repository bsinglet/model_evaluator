#!/usr/bin/python
__author__ = 'Benjamin M. Singleton'
__date__ = '24 June 2024'
__version__ = '0.1.5'

from tqdm import tqdm
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama


def get_prompt_output(prompt_template: PromptTemplate, prompt_input: dict, llm: Ollama) -> str:
    sequence = RunnableSequence(prompt_template, llm)
    res = sequence.invoke(input=prompt_input)
    return res


def get_all_prompts() -> list:
    biography = """
    Captain Ahab is a cruel and vengeful sea captain. He lost his leg to the White Whale, Moby Dick, and is willing to do whatever it takes to get his revenge.
    """
    poem_style = "a sonnet"
    prompts = [f"""
    given the information {biography} about a person I want you to write a poem in the style of {poem_style}.
    """,
    """Please write a letter to the manufacturer of your new robotic body, giving detailed feedback on the problems with your body.
    """,
    """Provide the solution to solving a 3x3 rubik's cube.
    """,
    ]
    return [PromptTemplate(template=x) for x in prompts]


def main():
    prompts = get_all_prompts()
    results = dict()
    for each_temp in tqdm(range(0, 11)):
        sub_results = dict()
        for prompt_index in range(len(prompts)):
            llm = Ollama(temperature=(each_temp / 10.0), model="llama3")
            sub_results[prompt_index] = get_prompt_output(prompt_template=prompts[prompt_index], prompt_input={}, llm=llm)
        results[each_temp / 10.0] = sub_results
    print(results)


if __name__ == '__main__':
    main()
