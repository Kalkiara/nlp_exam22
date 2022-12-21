import os
import openai
import torch
from transformers import pipeline, set_seed
import random
import csv

def load_tasks(file_name):
    """creates a list of prompts or sentences with [MASK]-tokens based on a txt-file

    Args:
        file_name (str): name of a .txt file containing the tasks given to LM model. 
                                Each line in the file should correspond to one task.

    Returns:
        list: a list with the tasks
    """
    data_path = os.path.join("data", file_name)
    with open(data_path) as f:
        lines = [line.rstrip() for line in f]
    
    return lines

def bert_test(model):
    
    """Text generation pipeline using BERT base and BERT large

    Args:
        model (str): name of the specific BERT model, either base or large 

    Returns:
        list: list of generated output words based on the prompts. 
    """
    list_tasks = load_tasks('tasks_bert.txt')
    outputs = []
    
    print("initializing pipeline and getting output")

    generator = pipeline('fill-mask', model = model)
    for task in list_tasks:
        set_seed(1999)
        temp_out = generator(task)
        outputs.append(temp_out[0]['token_str'])

    output = dict(zip(list_tasks, outputs))

    return output

def gpt_test(model):

    """Text generation pipeline using GPT-2 and 3

    Args:
        model (str): name of the specific GPT model, either 2 or 3 

    Returns:
        list: list of generated output words based on the prompts. 
    """
    list_tasks = load_tasks('tasks_gpt.txt')
    outputs = []

    print("initializing pipeline and getting output")

    if model == 'gpt2':
        generator = pipeline('text-generation', model = model, pad_token_id=50256)
        for task in list_tasks:
            set_seed(1999)
            temp_out = generator(task, max_new_tokens = 1)
            outputs.append(temp_out[0]['generated_text'].split(' ')[-1])

    elif model == 'gpt3':
        with open('api.txt') as f:
            openai.api_key = f.read()
        for task in list_tasks:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=task,
                temperature=0.6, 
                max_tokens=1,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=1,
                seed = 1999)

            outputs.append(response['choices'][0]['text'].strip())
    
    output = dict(zip(list_tasks, outputs))
    return output

def perform_test(model):
    """Test model knowledge of color using a list of tasks and a given LM. 

    Args:
        model (str): Name of an LM to test.

    Returns:
        dictionary or str: 
            if the specified model is one we have tested, a dictionary with tasks and the generated word for the given task is returned. 
            if the specified model is not one we have tested, a string saying this is returned. 
    """
    if model == 'gpt2' or model == "gpt3":
        output = gpt_test(model = model)        

    elif model == 'bert-base-uncased' or model == 'bert-large-uncased':
        output = bert_test(model = model)

    else:
        output = "the model you chose does not correspond to a model we've tested"

    return output

def save_output(output, output_filename):
    """saves output dictionary as a semi-colon separated .csv-file. 

    Args:
        output (dictionary): the output dictionary created using perform_tests()
        output_filename (str): the name of the output file. 
                               Has to end with .csv

    Returns:
        None (a file is saved in 'out' folder with the specified file name)
    """
    out_path = os.path.join("out", output_filename)

    with open(out_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter =";",quoting=csv.QUOTE_MINIMAL)
        for key, val in output.items():
            # write every key and value to file
            key = key.replace(' [MASK].', "")
            writer.writerow([key, val])

    return None 