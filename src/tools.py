import os
import openai
import torch
import transformers
from transformers import pipeline, set_seed
import random
import csv

def load_tasks(file_name):
    """_summary_

    Args:
        file_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_path = os.path.join("data", file_name)
    with open(data_path) as f:
        lines = [line.rstrip() for line in f]
    
    return lines

def gpt2_test(list_tasks):
    """_summary_

    Args:
        list_tasks (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        list_tasks (_type_): _description_

    Returns:
        _type_: _description_
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
    bert_base_pipe = pipeline('fill-mask', model='bert-base-uncased')
    outputs = []
    for task in list_tasks:
        set_seed(1999)
        temp_out = bert_base_pipe(task)
        outputs.append(temp_out[0]['token_str'])
    return outputs

def bert_large_test(list_tasks):
    bert_base_pipe = pipeline('fill-mask', model='bert-large-uncased')
    outputs = []
    for task in list_tasks:
        set_seed(1999)
        temp_out = bert_base_pipe(task)
        outputs.append(temp_out[0]['token_str'])
    return outputs

def perform_test(list_tasks, model):
    """_summary_

    Args:
        list_tasks (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
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


    

def save_output(output, output_filename):
    out_path = os.path.join("out", output_filename)

    with open(out_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter =";",quoting=csv.QUOTE_MINIMAL)
        for key, val in output.items():
        # write every key and value to file
            key = key.replace(' [MASK].', "")
            writer.writerow([key, val])