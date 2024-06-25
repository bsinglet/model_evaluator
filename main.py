#!/usr/bin/python
__author__ = 'Benjamin M. Singleton'
__date__ = '24 June 2024'
__version__ = '0.1.5'

from tqdm import tqdm
from langchain.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_community.llms import Ollama


def get_prompt_output(prompt_template: PromptTemplate, prompt_input: dict, temperatue: float) -> str:
    llm = Ollama(temperature=temperatue, model="llama3")
    
    sequence = RunnableSequence(prompt_template, llm)
    res = sequence.invoke(input=prompt_input)
    return res


def main():
    poem_template = """
    given the information {information} about a person I want you to write a poem in the style of {style}.
    """
    poem_prompt_template = PromptTemplate(input_variables=["information", "style"], template=poem_template)

    information = """
    Captain Ahab is a cruel and vengeful sea captain. He lost his leg to the White Whale, Moby Dick, and is willing to do whatever it takes to get his revenge.
    """
    results = dict()
    for each_temp in tqdm(range(0, 11)):
        results[each_temp / 10.0] = get_prompt_output(prompt_template=poem_prompt_template, prompt_input={"information": information, "style": "a sonnet"}, temperatue=(each_temp / 10.0))
    print(results)


if __name__ == '__main__':
    main()
