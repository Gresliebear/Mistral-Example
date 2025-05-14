# Mistral-Example

Example of using Mistral API and locally running the model

Create a account on Mistral https://auth.mistral.ai/ui/login 

#### You will mistral API key for app.py to work

create .env file

put this in the .env file


`MISTRAL_API_KEY="XXXXXXXXXXXXXXX"`

#### create a virtual environment

`python -m venv venv`

enter your virtual environment

`source venv/bin/activate` (Linux/MacOS)

or

`venv\Scripts\activate` (Windows)

#### Pip install the requirements

`pip install -r requirements.txt`

#### run the app

`python app.py`

#### Run the test script to past data flask api

`python test_script.py`

This should be able to give you good prototype test concept out

## Local usage 

you need to use Huggingface API key to download and access model weights

Huggingface tracks who is using the model they want maintain accountability for submissions and allow the community to identify the authors of models

Some of these models are not open source and require a license to use so always check the license before using a model

create a https://huggingface.co/join and create API key

`HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")`

All you need to do is find model you want and put its model tag below in gpu_app.py

`model_name = "mistralai/Mistral-7B-v0.3"`


### Hardware requirements 

Always check the model requirements for hardware it won't work if don't have enough memory or VRAM

So model Mistral-7B-v0.3 requires 
- FP16 ≈ 14 GB VRAM
- 4-bit ≈ 3.5 GB VRAM
- GPU: ≥ 12 GB (e.g. RTX 3060 12 GB); ≥ 16 GB for headroom
