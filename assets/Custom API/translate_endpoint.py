import flask, json, os, io
from flask import request
import requests
from dotenv import load_dotenv
from flask_cors import CORS
import pandas as pd

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference

load_dotenv()

app = flask.Flask(__name__)
app.secret_key = ".."
CORS(app)
app.config["DEBUG"] = True


#This will for now support translation between english, spanish, and japanese
example_dict = {"en": "Show Peer Risks", "es": "Mostrar riesgos de pares", "jp": "ピアリスクを表示", "ja": "ピアリスクを表示"}
lang_name_dict = {"en": "English", "es": "Spanish", "jp": "Japanese", "ja": "Japanese"}

#PROMPTS
lang_detect_prompt = """<|user|>
You will be given some text that is either English or Japanese, output in either "en" (for English) or "jp" (for Japanese), corresponding to the language of the given text. Do not output anything other than two letters.
Examples:
Text: Hello, how are you?
Output: en

Text: ピアリスクを表示
Output: jp

<|assistant|>
Text: {text}
Output: """

prompt = """<|user|>
You are an experienced {originLang}-{language} translator. You will be given an {originLang} word or phrase, and your task is to translate it into {language}. The output should be in a json format, with the only key being "translation" and its value being the translation. These words and phrases are coming from a corporate risk and compliance software, if that helps inform the context of these words and phrases. Do not output anything other than the {language} translation in a json format. Do not give an explanation for the translation, just give the json with the {language} characters.
- Try not to miss any information when text is converted into {language}, and try to maintain the same professional tone.
- If there are any complex vocabulary in {originLang}, feel free to convert that into any closest synonyms in {language}.
- Always write the output in a json format, and only write the translation in {language}.
Example: 
Input: {originExample}
Output: {langExample}
<|assistant|>
Input: {input}
Output: 
"""

history = []


PARAMS = {
        'min_new_tokens': 1,
        'max_new_tokens': 1000,
        'repetition_penalty': 1.0,
        'decoding_method': "greedy",
         "stop_sequences": [
            "}"
        ],
        "include_stop_sequence": True,
    }

DETECT_PARAMS = {
        'min_new_tokens': 1,
        'max_new_tokens': 3,
        'repetition_penalty': 1.0,
        'decoding_method': "greedy",
         "stop_sequences": [
            r'\n',
        ],
        "include_stop_sequence": False,
    }



WX_KEY = os.environ['WX_API_KEY']
WX_URL = os.environ['WX_API_URL']
WX_PROJECT_ID = os.environ["WX_PROJECT_ID"]
mistral_large_model_id = "mistralai/mistral-large"
mixtral_model_id = "mistralai/mixtral-8x7b-instruct-v01"

wx_creds = {
    "url": WX_URL,
    "apikey": WX_KEY,
}

def translate_wx(translatePrompt, model_id, params):

    model = ModelInference(
        model_id=model_id,
        credentials=wx_creds,
        params=params,
        project_id=WX_PROJECT_ID,
        space_id=None,
        verify=False,
    )

    response = model.generate_text(prompt=translatePrompt).replace("```json", "")

    return response

#retrieve dictionary from openpages, returns it as a dataframe
def get_dictionary() :
    url = os.environ["OP_DICT_URL"]
    auth = os.environ["OP_AUTH"]
    response = requests.get(url, headers={"Authorization": auth})
    return pd.read_excel(io.BytesIO(response.content), engine="openpyxl")

#Read their dictionary
dict_df = get_dictionary()#pd.read_excel("Custom_Dictionary.xlsx").rename(columns={"ja": "jp"})


#This is the main route
@app.route("/translate", methods=["POST"], strict_slashes=False)
def translate_ns() :
    body = request.get_json()
    try :
        texts = body['text']
        targetLang = body['target'].replace("ja", "jp") #bandaid, openpages shortens japanese to ja
    except:
        return "Invalid input format", 402
    
    try:
        targetExample = example_dict[targetLang]
    except:
        return "Language not supported", 403
    
    model_id = mistral_large_model_id
    if "model_id" in body.keys() :
        model_id = body["model_id"]
    
    translations = []
    translation  = ""
    #If there is more than one text translate all of them, openpages sends fields one at a time
    for text in texts: 

        #Origin language detection
        completedPrompt = lang_detect_prompt.format(text=text)
        
        lang = translate_wx(completedPrompt, mixtral_model_id, DETECT_PARAMS).replace(" ", "")
        if lang not in ["en", "jp"] :
            lang = "en"

        originExample = example_dict[lang]

        #Check if its in the given dictionary
        if text in dict_df[lang].values :
            translation = dict_df.loc[dict_df[lang] == text][targetLang].iloc[0]
            print(translation)
        else : #Run the LLM to translate using one example
            completedPrompt = prompt.format(originLang=lang_name_dict[lang], 
                                            language=lang_name_dict[targetLang],
                                            originExample=originExample,
                                            langExample=targetExample,
                                            input=text)
            try:
                translation = json.loads(translate_wx(completedPrompt, model_id, PARAMS))['translation']
            except :
                translation = ""
                print("oof :(")
        translations.append(translation)
    
    output = {"translations": translations, "word_count": sum([t.count(' ') + 1 for t in translations]), "character_count": sum([len(t) for t in translations]), "detected_language": lang, "detected_language_confidence": 1}
    return output, 200



#for debugging remotely
@app.route("/history")
def get_history() :
    global history
    return history, 200

@app.route("/clear")
def clear() :
    global history
    history = []
    return [], 200


#LOGGING
@app.after_request
def save_response(r):
    global history
    if "/history" not in flask.request.url :
        history.append([
            request.data.decode('utf-8'),
            r.get_data().decode('utf-8')
        ])
    return r

@app.before_request
def before() :
    global history
    history.append([
        flask.request.url
    ])
