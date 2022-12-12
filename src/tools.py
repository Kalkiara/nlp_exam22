import os
import openai
import torch
from transformers import pipeline, set_seed
import random
import csv

def load_tasks(file_name):
    """creates a list of prompts or sentences with [MASK]-tokens based on a txt-file

    Args:
        file_name (.txt file): .txt file containing the tasks given to LM model. 
                                Each line should correspond to one task.

    Returns:
        list: a list with the tasks
    """
    data_path = os.path.join("data", file_name)
    with open(data_path) as f:
        lines = [line.rstrip() for line in f]
    
    return lines

def gpt2_test(task_file):
    """Text generation pipeline using GPT-2

    Args:
        task_file (str): name of txt-file containing the tasks. 
                         Defaults to our GPT prompt tasks. 
    Returns:
        list: list of generated output words based on the prompts. 
    """
    list_tasks = load_tasks(task_file)
    print("initializing pipeline and getting output")
    gpt2_generator = pipeline('text-generation', model = 'gpt2', pad_token_id=50256)

    outputs = []
    for task in list_tasks:
        set_seed(1999)

        temp_out = gpt2_generator(task, max_new_tokens = 1)

        outputs.append(temp_out[0]['generated_text'].split(' ')[-1])
    output = dict(zip(list_tasks, outputs))

    return output

def gpt3_test(task_file):
    """Text generation pipeline using GPT-3

    Args:
        task_file (str): name of txt-file containing the tasks. 
                         Defaults to our GPT prompt tasks. 

    Returns:
        outputs (list): list of generated output words based on the prompts.
    """
    with open('api.txt') as f:
        openai.api_key = f.read()

    list_tasks = load_tasks(task_file)

    outputs = []
    print("initializing model and getting output")

    for task in list_tasks:
        response = openai.Completion.create(
                model="text-davinci-003",
                prompt=task,
                temperature=0.4, #changed from 0.6, FIX AND TEST
                max_tokens=1,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=1,
                seed = 1999
                #logprobs = 5 # FIX THIS FOR OTHER MODELS AS WELL SO OUTPUT IS SAVED PROPERLY
                )
        outputs.append(response['choices'][0]['text'].strip())
    output = dict(zip(list_tasks, outputs))

    return output

def bert_base_test(task_file):
    """Fill masked token pipeline using BERT base, uncased. 

    Args:
        task_file (str): name of txt-file containing the tasks. 
                         Defaults to our BERT masked token tasks. 
    Returns:
        list: list of generated tokens for each [MASK] in each task.
    """
    list_tasks = load_tasks(task_file)
    print("initializing pipeline and getting output")

    bert_base_pipe = pipeline('fill-mask', model='bert-base-uncased')
    outputs = []
    for task in list_tasks:
        set_seed(1999)
        temp_out = bert_base_pipe(task)
        outputs.append(temp_out[0]['token_str'])
    output = dict(zip(list_tasks, outputs))

    return output

def bert_large_test(task_file):
    """Fill masked token pipeline using BERT large, uncased. 

       Args:
        task_file (str): name of txt-file containing the tasks. 
                         Defaults to our BERT masked token tasks. 
    Returns:
        list: list of generated tokens for each [MASK] in each task.
    """
    list_tasks = load_tasks(task_file)
    print("initializing pipeline and getting output")

    bert_large_pipe = pipeline('fill-mask', model='bert-large-uncased')
    outputs = []
    for task in list_tasks:
        set_seed(1999)
        temp_out = bert_large_pipe(task)
        outputs.append(temp_out[0]['token_str'])
    output = dict(zip(list_tasks, outputs))

    return output

def perform_test(model, file_name):
    """Test model knowledge of color using a list of tasks and a given LM. 

    Args:
        list_tasks (list): list of tasks created using load_tasks().
        model (str): Name of an LM to test. Defaults to GPT-2

    Returns:
        dictionary or str: 
            if the specified model is one we have tested, 
            a dictionary with tasks and the generated word for the given task is returned. 
            if the specified model is not one we have tested, a string saying this is returned. 
    """
    if model == 'gpt2':
        output = gpt2_test(task_file = file_name)
        
    elif model == 'gpt3':
        output = gpt3_test(task_file = file_name)        

    elif model == 'bert_base':
        output = bert_base_test(task_file = file_name)

    elif model == 'bert_large':
        output = bert_large_test(task_file = file_name)

    else:
        output = "the model you chose does not correspond to a model we've tested"

    return output

def save_output(output, output_filename):
    """saves output dictionary as a semi-colon separated .csv-file. 

    Args:
        output (dictionary): the output dictionary created using perform_tests()
        output_filename (str): the name of the output file. 
                               Has to end with .csv, defaults to 'output.csv'

    Returns:
        None (instead, a file is saved in 'out' folder with the specified name)
    """
    out_path = os.path.join("out", output_filename)

    with open(out_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter =";",quoting=csv.QUOTE_MINIMAL)
        for key, val in output.items():
        # write every key and value to file
            key = key.replace(' [MASK].', "")
            writer.writerow([key, val])

    return None 