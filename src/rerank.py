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

def get_prefix_prompt(query, instructions, num):
    rank_instruction = "Rank and Select the most relevant English recipes based on their relevance to the given Chinese recipe,"
    rank_instruction += "When the relevance is the same, prioritize recipes that are more aligned with the culture of English speakers."
    return [
            {'role': 'system',
             'content': f"For {num} English recipes, each indicated by number identifier in []. {rank_instruction}"
            },
            {'role': 'system',
             'content': f"relevance criteria : {instructions}"
            },
            {'role': 'user',
             'content': f"Chinese recipe: {query}"
            }
        ]

def get_post_prompt(num):
    response = f"Rank the {num} English recipe above based on their relevance and cultural adaptability."
    response += " Select the identifier of the Best English Recipe, like [0]."
    response += " Please only report the identifier of the most relevant English recipe, without any other words or explaination."
    return  response 

def create_permutation_instruction(query, num, documents, relevance_instructions, max_length):
    
    messages = get_prefix_prompt(query, relevance_instructions, num)
    rank = 0
    for document in documents: 
        content = document.strip()
        content = ' '.join(content.split()[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        rank += 1
    messages.append({'role': 'user', 'content': get_post_prompt(num)})

    return messages

def request_llm_rerank(messages, model_id):
    global llm_tokenizer, llm_model
    if llm_tokenizer is None or llm_model is None:
        load_llm_model(model_id)
     
    input_ids = llm_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    #terminators = [llm_tokenizer.eos_token_id, llm_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = llm_model.generate(input_ids, max_new_tokens=256)#, eos_token_id=terminators)
    response = outputs[0][input_ids.shape[-1]:]
    response = llm_tokenizer.decode(response, skip_special_tokens=True)
    try:
        id = response.split(']')[0].split('[')[1]
        id = int(id)
    except:
        id = 0
        print (f"Error in rerank, response: {response}")
    return id

def rerank_top1(documents, query, ingredient, step, model_id, relevance_instructions, max_length):
    rerank_query = "标题: " + query +"\t" + "原料: " + ingredient + "\t" + "步骤: " + step
    rerank_query = rerank_query[:max_length]
    messages = create_permutation_instruction(rerank_query, len(documents), documents, relevance_instructions, max_length)
    id = request_llm_rerank(messages, model_id)
    try:
        document = documents[id]
    except:
        document = documents[0]
        print (f"Error in rerank, id: {id}")
    return document
            



