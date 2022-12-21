# nlp_exam22

# Summary

This repository is used for generating reproducible outputs from `GPT-2`, `GPT-3`, `BERT base`, and `BERT large` in terms of intrinsic color knowledge from a predefined set of diagnostics. See the corresponding exam paper.

## Performance

#### Overview of model performance
Language model| GPT-2 | GPT-3 | BERT base | BERT large | 
--- | --- | --- | --- |--- |
Percentage colors provided | 22.2% | 59.3% | 51.9% | 48.1% |
Percentage correct completions | 7.4% | 55.6% | 3.7% | 18.5% |

csv files containing the predictions of each model on each task can also be found under the folder 'out', see below

## Project Organization
The organization of the project is as follows:


```
├── README.md                <- The top-level README for this project
├── data                     <- Folder containing predefined color-specific tasks for the models to be tested on
|   ├── tasks_bert.txt       <- Tasks formatted for BERT models
|   └── tasks_gpt.txt        <- Tasks formatted for GPT models
├── out                      <- Folder containing outputs from the tasks as csv files
|   ├── output_bert.csv      <- csv file containing outputs from BERT base
|   ├── output_bert_l.csv    <- csv file containing outputs from BERT large
|   ├── output_gpt2.csv      <- csv file containing outputs from GPT2
|   └── output_gpt3.csv      <- csv file containing outputs from GPT3
├── src                      <- The main folder for scripts
|   ├── tools.py             <- A script containing functions used for loading tasks and performing masked word prediction for GPT2, GPT3, BERT base and BERT large 
|   └── main.py              <- A script containing the main function to access masked word prediction across models
├── api.txt                  <- Empty txt file for your personal OpenAI API key
├── .gitignore               <- A list of files not uploaded to git
├── requirements.txt         <- A requirements file specifying the required packages
└── run.sh                   <- Script to set up a virtual environment with the requirements from requirements.txt and run main.py 
```

## Reproducibility
You can run and reproduce results of word prediction across models by cloning the GitHub repository, generating an OpenAI access API key, and running the command line codes as provided below.

#### Run the following in the terminal: 
```
sudo apt-get update
sudo apt-get install python3-venv
```
#### Generate and set up OpenAI API key

Generate your OpenAI API key at
```
https://beta.openai.com/account/api-keys
```
Paste your personal API key in the txt file called "api.txt" 

#### Run code
Code should be run from the terminal sticking to the following structure:

```
bash run.sh --language_model --output_filename
```
#### Parameter explanation:
```
--language_model (-lm): The name of the language model 
(Options: gpt2, gpt3, bert-base-uncased, bert-large-uncased)

--output_filename (-o): The name of the output file. Should end with the .csv extension 
```
## Reproduce the results

Copy and run the following to reproduce our findings for each of the language models:

`GPT-2`  
```
bash run.sh gpt2 output_gpt2.csv
```
`GPT-3`  
```
bash run.sh gpt3 output_gpt3.csv
```
`BERT base`
```
bash run.sh bert-base-uncased output_bert.csv
```
`BERT large`
```
bash run.sh bert-large-uncased output_bert_l.csv
```
