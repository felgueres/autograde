import argparse
import os
from gpt import gpt
from functools import partial
import json
from tasks.extract import ExtractTask
from tasks.population_span_extraction import PopulationSpanExtractionTask

def get_task(name, file=None):
    if name == 'extract':
        return ExtractTask(file)
    if name == 'population_span_extraction':
        return PopulationSpanExtractionTask(file)
    else:
        raise NotImplementedError

def get_samples(task, x, y, n_generate_sample, stop):
    prompt = task.standard_prompt_wrap(x,y)
    samples = gpt(prompt, n=n_generate_sample, stop=stop) # generate one message for chat model 
    return [y + _ for _ in samples]

def solve(args, task, idx):
    x = task.get_input(idx)
    ys = get_samples(task, x, '', args.n_generate_sample, stop=None)
    return ys, {}

def run(args):
    task = get_task(args.task, args.task_file_path)
    logs,cnt_avg,cnt_any = [],0,0

    global gpt 
    gpt = partial(gpt, model=args.model, temperature=args.temperature)

    # TODO: add save to db
    file = f'logs/{args.task}/{args.model}_{args.method_generate}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_idx, args.task_end_idx):
        ys,info = solve(args, task, i)
        infos = [task.test_output(i,y) for y in ys]
        info.update({'idx': i, 'ys': ys, 'infos': infos})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)

        # TODO: add compute metrics
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(f'Finished running task {i} with accuracy {sum(accs) / len(accs)}')
        print(f'Summary: {json.dumps(info, indent=2)}')
    
    n = args.task_end_idx - args.task_start_idx
    print(f'Finished running {n} tasks.')

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=['gpt-3.5-turbo', 'gpt-4'], default='gpt-3.5-turbo')
    args.add_argument('--temperature', type=float, default=0.5)
    args.add_argument('--task', type=str, choices=['extract', 'population_span_extraction'])
    args.add_argument('--task_file_path', type=str, required=True)
    args.add_argument('--task_start_idx', type=int, default=0)
    args.add_argument('--task_end_idx', type=int, default=1)
    args.add_argument('--method_generate', type=str, choices=['sample'])
    # TODO: Add method_evaluate to support evaluation-based prompting techinques, eg. voting, scoring 
    # TODO: Add method_retrieve to support retrieval-based prompting techinques, eg. top-k, nearest neighbor 
    args.add_argument('--n_generate_sample', type=int, default=1)
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    run(args)