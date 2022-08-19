# _Python Assignment 1 for GreyOrange Training_
## GME-410
> Problem Statement : Identify a reasonable "chat" data set and build a simple Python application to process the data, generate an Emoji suggestion tool (using a simple sentiment analysis model of your choice) which classifies the text and returns an appropriate Emoji.  The application should have a very simple interface for the user to write a message to it and receive an Emoji as output. 


**This assignment covers the implementation of a Emotion Classification based on the input text message by the user**



### Following are 3 main components of this project
- A Streamlit based Interface to interact with trained models and analyse the outputs
- A trained DL model [ LSTM ] 
- A traind ML Model [ Logistic Regression ]

## Features

- You can get the probability of the emotion label predicted
- The emotions are also represented in the form of Emojis
- Option to train both ML or DL Model
- Streamlit application provides an easy to use dashboard for the users
- You can test your input sentences with both ML and DL algortihms

## Setup the project on your system
`Step1`:  ```$ git clone https://github.com/guneetsk99/ProjectAssignment1GO.git```

`Step2`:  ```$ pip install -r requirements.txt```

`Step3`: `$ python3 main.py --help `
>Emotion Based Text Classification
>optional arguments:
>  -h, --help            show this help message and exit
  
 > --train_data_csv 
                        Enter path of training data CSV
                        
 > --training_type 
                        Enter 0 for ML and 1 for DL
                        
  >--run_demo   Enter 1 to run Demo or any other number for training
  
  
`Step3`: To use pretarined weights and access the streamlit dashboard
`$ python3 main.py --run_demo 1`

`Step4`: Retrain the ML model on the input data
`$ python3 main.py --train_data_csv <your_input_csv> --training_type 0`

`Step5`: Retrain the DL model on the input data
`$ python3 main.py --train_data_csv <your_input_csv> --training_type 1`

