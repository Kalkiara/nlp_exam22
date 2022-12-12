# nlp_exam22

# Summary

This repository is used for generating reproducible outputs from GPT2, GPT3, BERT base, and BERT large in terms of intrinsic color knowledge from a predefined set of diagnostics. See the corresponding exam paper.

# Performance
xxx do we want to highlight our output in table format here? 

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
├── .gitignore               <- A list of files not uploaded to git
├── requirements.txt         <- A requirements file specifying the required packages
└── run.sh                   <- Script to set up a virtual environment with the requirements from requirements.txt and run main.py 


```

## Running the code
You can run and reproduce results of masked word prediction across models by cloning the GitHub repository, generating an OpenAI access API key, and running the command line code as provided below.

### Generate and set up OpenAI API key
```
Generate your OpenAI API key at https://beta.openai.com/account/api-keys
Paste your personal API key in the txt file called "api.txt"

```
### Run code:
```
bash run.sh xxx something something

```
### Parameter explanation:
```
something: blablabla
other_something: blablabal

```
### To reproduce our findings:
```
bash run.sh xxx something_specific something_specific

```