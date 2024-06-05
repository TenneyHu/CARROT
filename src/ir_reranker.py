import argparse
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from data_loader import instruction_loader

def get_prefix_prompt(query, instructions, num):
    return [
        {'role': 'system',
             'content': instructions},
        {'role': 'user',
             'content': f"For {num} English recipes, each indicated by number identifier in []. Rank English recipes based on their relevance to the Chinese recipe: {query}."},
        ]

def get_post_prompt():
    reponse = " The recipes should be listed in descending order using identifiers. The most relevant recipes should be listed first. "
    reponse += " The output format should be [] > [] > ... > [] , e.g., [1] > [2] > [3]."
    reponse += " should list exactly ten different most relevant recipes, one identifier should not appear more than once."
    reponse += " Please only report the ten different identifier of the most relevant English recipe, without any other words or explaination."
    return reponse 

def create_permutation_instruction(query, num, documents, instructions):
    
    messages = get_prefix_prompt(query, instructions, num)
    rank = 0
    for document in documents: 
        messages.append({'role': 'user', 'content': f"[{rank}] {document}"})
        rank += 1
    messages.append({'role': 'user', 'content': get_post_prompt()})
    return messages

def parse_list(arg_value):
    return [item.strip() for item in arg_value.split(',') if item.strip()]

def load_retrieval_results(results_pool):
    results = defaultdict(list)
    for result_file in results_pool:
        with open(result_file, 'r') as f:
            for lines in f:
                line = lines.strip().split('\t')
                qid = line[1]
                docid = line[2]
                results[qid].append(docid)
    return results

def chinese_recipes_loader(filepath, max_length=300):
    qid = 0
    chinese_recipes = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            text =  "标题: " + line[0] +" 原料: " + line[1] + " 步骤: " + line[2]
            text = text[:max_length]
            chinese_recipes[str(qid)] = text
            qid += 1
    return chinese_recipes

def english_recipes_loader(filepath, max_length=300):
    recipes = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            try:
                id = line[0]
                title = line[1]
                ingredients = line[2]
                steps = line[3]
                text = "Title: " + title + " Ingredients: " + ingredients + " Steps: " + steps
                text = text[:max_length]
                recipes[id] = text
            except:
                pass
    return recipes

class CARROT_IR_Reranker:
    def __init__(self, results, chinese_recipes, english_recipes, instructions):
        self.results_pool = results
        self.instructions = instructions
        self.chinese_recipes = chinese_recipes
        self.english_recipes = english_recipes

    def llm_reranker(self):

        api_token = "" #your api token
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token = api_token)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token = api_token
        )

        with open(args.output, 'w') as f:
            for qid in tqdm(self.results_pool):
                chinese_recipe = self.chinese_recipes[qid]
                english_recipes = []
                recipes_id = {}
                for i,docid in enumerate(self.results_pool[qid]):
                    english_recipes.append(self.english_recipes[docid])
                    recipes_id[i] = docid

                messages = create_permutation_instruction(chinese_recipe, len(english_recipes), english_recipes, self.instructions)

                input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)

                outputs = model.generate(input_ids,max_new_tokens=256)
                response = outputs[0][input_ids.shape[-1]:]
                response = tokenizer.decode(response, skip_special_tokens=True)
                print (response)
                for x in response.split('>'):
                    result = int(x.strip().split('[')[1].split(']')[0])
                    if result in recipes_id:
                        docid = recipes_id[result]
                        f.write(f"-1\t{qid}\t{docid}\n")
                    else:
                        print (f"Error: {result}")
                        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CARROT IR')
    parser.add_argument('--max_length', type=int, default=512, help='text max length')
    parser.add_argument('--output', type=str, default="./IR_Dataset/results/carrot.res", help='output file')
    parser.add_argument('--chinese_recipes', type=str, default="./ChineseRecipes/ir.tsv", help='path to chinese recipe file, the text should be title \t ingredients \t steps')
    parser.add_argument('--english_recipes', type=str, default="./EnglishRecipes/english_recipes.tsv", help='English Recieps DataBase tsv, the text should be index \t title \t ingredients \t steps')
    parser.add_argument('--relevance_instructions', type=str, default="./Instructions/relevance_instruction", help='rerank relevance instructions')
    parser.add_argument('--results_pool', type=parse_list, default="", help='retrieve results')
    
    args = parser.parse_args()
    english_recipes = english_recipes_loader(args.english_recipes, args.max_length)
    chinese_recipes = chinese_recipes_loader(args.chinese_recipes, args.max_length)
    results  = load_retrieval_results(args.results_pool)
    instructions = instruction_loader(args.relevance_instructions)
    carrot_ir = CARROT_IR_Reranker(results, chinese_recipes, english_recipes, instructions)
    carrot_ir.llm_reranker()