# autograde

Lightweight framework to evaluate LLM using model scoring methods

![flow.png](flow.png)

---

## Features

### Task-oriented

autograde has a simple interface to define tasks, which helps to easily bring your real-world applications. 

```py
from autograde.tasks.base import Task

class MyTask(Task):
    '''Task classes override two simple methods
    '''
    def get_input()
        # logic to inputs 

    def test_output()
        # logic to evaluate model outputs
```

Gets task type and data from args 
| -- /data/sample.jsonl 
Runs model task 
| -- Get task logic from /tasks/<task_type>.py
| -- Get prompt from /prompts/<task_type>.yaml
Gets model grader 
| -- Get prompt from config /graders/<grader_type>.yaml
Grade the model output
Save metrics

### Decoupled sampling from scoring 

autograde is focused on using models to score outputs. The model used for sampling and evaluating can be different, eg. you can use a bigger model at sampling and smaller for evals. 

## Quickstart 

For a quick demo, install requirements:

```sh
pip install -r requirements.txt
```

And simply run:

```sh
python run.py \
    --task extract \
    --task_file_path population_span_extraction.jsonl \
    --method_retrieve **top_k** \
    --method_generate sample \
    --temperature 0.5 \
    --task_start_idx 0 \
    --task_end_idx 1
```

Add examples
