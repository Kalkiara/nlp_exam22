import os
import openai
import torch
import transformers
from transformers import pipeline, set_seed
import random
import csv

def load_tasks(file_name):
    """creates a list of prompts or sentences with [MASK]-tokens based on a txt-file

    Args:
        file_name (.txt file): .txt file containing the tasks given to LM model. 
                                Each line should correspond to one task

    Returns:
        a list 
    """
    data_path = os.path.join("data", file_name)
    with open(data_path) as f:
        lines = [line.rstrip() for line in f]
    
    return lines

def gpt2_test(list_tasks):
    """Text generation pipeline using GPT-2

    Args:
        list_tasks (list): list of tasks created using load_tasks(). 

    Returns:
        list: list of generated output words based on the prompts. 
    """
    gpt2_generator = pipeline('text-generation', model = 'gpt2')
    outputs = []
    for task in list_tasks:
        set_seed(1999)

        temp_out = gpt2_generator(task, max_new_tokens = 1)

        output_word = temp_out[0]['generated_text'].split(' ')[-1]

        outputs.append(output_word)

    return outputs

def gpt3_test(list_tasks):
    """Text generation pipeline using GPT-3

    Args:
        list_tasks (list): list of tasks created using load_tasks().

    Returns:
        outputs (list): list of generated output words based on the prompts.
    """
    openai.api_key = "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"

    outputs = []
    for task in list_tasks:
        set_seed(1999)

        response = openai.Completion.create(
                model="text-davinci-003",
                prompt=task,
                temperature=0.6,
                max_tokens=1,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=1
                )
        outputs.append(response['choices'][0]['text'].strip())

    return outputs 

def bert_base_test(list_tasks):
    """Fill masked token pipeline using BERT base, uncased. 

    Args:
        list_tasks (list): list of tasks created using load_tasks().

    Returns:
        list: list of generated tokens for each [MASK] in each task.
    """
    bert_base_pipe = pipeline('fill-mask', model='bert-base-uncased')
    outputs = []
    for task in list_tasks:
        set_seed(1999)
        temp_out = bert_base_pipe(task)
        outputs.append(temp_out[0]['token_str'])
    return outputs

def bert_large_test(list_tasks):
    """Fill masked token pipeline using BERT large, uncased. 

    Args:
        list_tasks (list): list of tasks created using load_tasks().

    Returns:
        list: list of generated tokens for each [MASK] in each task.
    """
    bert_base_pipe = pipeline('fill-mask', model='bert-large-uncased')
    outputs = []
    for task in list_tasks:
        set_seed(1999)
        temp_out = bert_base_pipe(task)
        outputs.append(temp_out[0]['token_str'])
    return outputs

def perform_test(list_tasks, model = 'gpt2'):
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
        outputs = gpt2_test(list_tasks)
        
        output = dict(zip(list_tasks, outputs))

    elif model == 'gpt3':
        outputs = gpt3_test(list_tasks)
        output = dict(zip(list_tasks, outputs))
        

    elif model == 'bert_base':
        outputs = bert_base_test(list_tasks)
        output = dict(zip(list_tasks, outputs))

    elif model == 'bert_large':
        outputs = bert_large_test(list_tasks)
        output = dict(zip(list_tasks, outputs))

    else:
        output = "the model you chose does not correspond to a model we've tested"

    return output

def save_output(output, output_filename = 'output.csv'):
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