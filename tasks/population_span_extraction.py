# Contains the ExtractTask class, used to extract facts from documents.
from .base import Task
import re
import os
import json
from gpt import gpt 
import yaml

DATA_PATH = './data'

class PopulationSpanExtractionTask(Task):
    '''
    Inputs: 
        (a) : a text from a document 
        (b) : a key phrase used to extract
    Output: 
        (y) : a key-value pair extracted from (a) where the key is (b) 
    Input Example:
        (a) : "The quick brown fox jumps over lazy dog"
        (b) : "fox_color"
    Output Example:
        (y) : {{'fox_color': 'brown'}} 
    '''
    MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y), }

    def __init__(self, file):
        super().__init__()
        path = os.path.join(DATA_PATH, file)
        self.data = open(path, 'r').readlines()
        self.stops = [None]
    
    def get_input(self, idx: int) -> str:
        return json.loads(self.data[idx])['input']
    
    def get_ideal(self, idx:int) -> str:
        return json.loads(self.data[idx])['ideal']
    
    def test_output(self, idx: int, output: str): 
        import string
        match_fn = self.MATCH_FNS['starts_or_endswith']
        choices = ['A', 'B', 'C', 'D', 'E']
        prompt = self.eval_prompt_wrap(x=self.get_input(idx),y_pred=output, ideal=self.get_ideal(idx), type='fact')
        score_outputs = gpt(prompt, n=1, model='gpt-3.5-turbo')
        scores = []
        r = 0
        for score_output in score_outputs:
            print('Model scorer: \n********\n', score_output, '\n********\n')
            lines = score_output.split('\n')
            for line in lines:
                line = line.strip()
                line = "".join(c for c in line if c not in string.punctuation)
                if not line: continue
                for choice in choices:
                    if match_fn(line, choice):
                        scores.append(choice)
                        r += 1 if choice != 'D' else 0
                        break
        info = {'rs': scores, 'r': r} 
        return info
    
    @staticmethod
    def standard_prompt_wrap(x:dict, y:str='') -> str:
        extract_config = yaml.safe_load(open('prompts/population_span_extraction.yaml', 'r'))
        standard_prompt = extract_config['extract']['prompt']
        return standard_prompt.format(text=x[1]['content']) + y
    
    # Implement fetching evaluation prompt from yaml
    @staticmethod
    def eval_prompt_wrap(x, y_pred: str, ideal: str, type='fact') -> str:
        grader_config = yaml.safe_load(open(f'graders/{type}.yaml', 'r'))
        eval_prompt = grader_config['fact']['prompt'].format(input=' '.join([_['content'] for _ in x]), ideal=ideal, completion=y_pred) 
        return eval_prompt 