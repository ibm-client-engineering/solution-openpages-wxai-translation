import xml.etree.ElementTree as ET 
import re, os, requests, io
import pandas as pd

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference


from dotenv import load_dotenv
load_dotenv()

xmlfile = "XML FILE PATH GOES HERE"
jp_regex = '[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]'

WX_KEY = os.environ['WX_API_KEY']
WX_URL = os.environ['WX_API_URL']
WX_PROJECT_ID = os.environ["WX_PROJECT_ID"]
mistral_large_model_id = "mistralai/mistral-large"
mixtral_model_id = "mistralai/mixtral-8x7b-instruct-v01"

wx_creds = {
    "url": WX_URL,
    "apikey": WX_KEY,
}

TRANSLATION_PROMPT = """<|user|>
You are an experienced {originLang}-{language} translator. You will be given an {originLang} word or phrase, and your task is to translate it into {language}. The output should be in a json format, with the only key being "translation" and its value being the translation. These words and phrases are coming from a corporate risk and compliance software, if that helps inform the context of these words and phrases. Do not output anything other than the {language} translation in a json format. Do not give an explanation for the translation, just give the json with the {language} characters.
- Try not to miss any information when text is converted into {language}, and try to maintain the same professional tone.
- If there are any complex vocabulary in {originLang}, feel free to convert that into any closest synonyms in {language}.
- Always write the output in a json format, and only write the translation in {language}.
Example: 
Input: {originExample}
Output: {langExample}
<|assistant|>
"""


PARAMS = {
        'min_new_tokens': 1,
        'max_new_tokens': 500,
        'repetition_penalty': 1.0,
        'decoding_method': "greedy",
         "stop_sequences": [
            '\n',
            '---'
        ],
        "include_stop_sequence": False,
    }


def get_prompt(text_block):
    prompt_template = TRANSLATION_PROMPT + f"Input: {text_block}\n Output: "
    return prompt_template


def translate_wx(text_block, model_id):

    model = ModelInference(
        model_id=model_id,
        credentials=wx_creds,
        params=PARAMS,
        project_id=WX_PROJECT_ID,
        space_id=None,
        verify=False,
    )

    response = model.generate_text(prompt=get_prompt(text_block)).replace("```json", "")

    try: 
        result = response.json()["results"][0]["generated_text"]
        result = result.replace("\n", "\\n")
        result = result.strip("\\n")
        return result
    except:
        return text_block

#retrieve dictionary from openpages, returns it as a dataframe
def get_dictionary() :
    url = os.environ["OP_DICT_URL"]
    auth = os.environ["OP_AUTH"]
    response = requests.get(url, headers={"Authorization": auth})
    return pd.read_excel(io.BytesIO(response.content), engine="openpyxl")


def is_jp(text) :
    return re.search(jp_regex, text)

#get jp-en dictionary
dict_df = get_dictionary()

# create element tree object 
tree = ET.parse(xmlfile) 
  
# get root element 
root = tree.getroot() 
print(root)


jp_node = None
for child in root.findall("./applicationStrings/localeStrings") :
    print(child.attrib['name'])
    if child.attrib['name'] == "Japanese" :
        jp_node = child

for jp_child in jp_node :
    text = jp_child.attrib['value']
    if is_jp(text) :
        continue
    elif (len(text) > 0): 
        #Check if its in the given dictionary
        if text in dict_df['en'].values :
            translation = dict_df.loc[dict_df['en'] == text]['ja'].iloc[0]
        else : #Run the LLM to translate using one example
            j_text = translate_wx(text, mistral_large_model_id)
            jp_child.attrib['value'] = j_text
                

tree.write(xmlfile)