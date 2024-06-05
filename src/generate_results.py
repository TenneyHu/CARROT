import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
llm_tokenizer = None
llm_model = None

def load_llm_model(model_id):
    global llm_tokenizer, llm_model
    api_token = "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"
    llm_tokenizer = AutoTokenizer.from_pretrained(model_id, token=api_token, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token = api_token, trust_remote_code=True)

def get_prompt(query):
    templete = "Convert the provided Chinese recipe into an English recipe"
    templete += " so that it fits within Western cooking culture, "
    templete += " with Western cooking knowledge, and meets is consistent a Western recipe’s style."
    templete += " The output format should be: Title: [English Recipe Title] + Ingredients: [English Recipe Ingredients] + Steps: [English Recipe Steps]"
    templete += " Please only output the title, ingredients, and steps in English, do not add any other content."
    return [
        {'role': 'system',
            'content': f"{templete}"
        },
        {'role': 'user',
            'content': f"Chinese recipe: {query}"
        }
    ]

def request_llm_generation(messages, model_id):
    global llm_tokenizer, llm_model
    if llm_tokenizer is None or llm_model is None:
        load_llm_model(model_id)
     
    input_ids = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    #terminators = [llm_tokenizer.eos_token_id, llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = llm_model.generate(input_ids, max_new_tokens=256)#, eos_token_id=terminators)
    response = outputs[0][input_ids.shape[-1]:]
    response = llm_tokenizer.decode(response, skip_special_tokens=True)
    return response

def generation(query, ingredient, step, model_id, max_length):
    generation_query = "标题: " + query +"\t" + "原料: " + ingredient + "\t" + "步骤: " + step
    generation_query = generation_query[:max_length]
    messages = get_prompt(generation_query)
    return request_llm_generation(messages, model_id)


            



