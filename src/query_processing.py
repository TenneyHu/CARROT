from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
translate_tokenizer = None
translate_model = None

llm_tokenizer = None
llm_model = None

def load_translate_model(model_id):
    global translate_tokenizer, translate_model
    translate_tokenizer = AutoTokenizer.from_pretrained(model_id)
    translate_model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

def load_llm_model(model_id):
    global llm_tokenizer, llm_model
    api_token = "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"
    llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=api_token)
    llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token = api_token)

def translate_query(query, model_id):
    global translate_tokenizer, translate_model
    if translate_tokenizer is None or translate_model is None:
        load_translate_model(model_id)

    tokenized_text = translate_tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(device)
    translation = translate_model.generate(**tokenized_text)
    translated_text = translate_tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text

def request_llm(templete, request, model_id):
    global llm_tokenizer, llm_model
    if llm_tokenizer is None or llm_model is None:
        load_llm_model(model_id)
    messages = [{"role": "system", "content": templete}, {"role": "user", "content": request}]   
    input_ids = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    terminators = [llm_tokenizer.eos_token_id, llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = llm_model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=False)
    response = outputs[0][input_ids.shape[-1]:]
    response = llm_tokenizer.decode(response, skip_special_tokens=True)
    return response

def recipe_query_generation(ingredient, step, model_id):
    templete = "Here is a Chinese recipe; please create a short English title for the recipe."
    request =  "Ingredients: " + ingredient + " Steps: " + step  + " Please only output the title, do not add any other content."
    return request_llm(templete, request, model_id)

def query_adaptation(title, model_id):
    templete = "This is a Chinese recipe title, rewritten to fit English cultural conventions"
    request = "Title: " + title + " Please only output the english recipe title, do not add any other content."
    return request_llm(templete, request, model_id)