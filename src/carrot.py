import argparse
from tqdm import tqdm
from query_processing import translate_query, recipe_query_generation, query_adaptation
from data_loader import chinese_recipes_loader, english_recipes_loader, instruction_loader
from retrieve import get_retrieve_results
from rerank import rerank
from generate_results import generation

class Carrot:
    def __init__(self, args):
        self.args = args
        self.english_recipes = english_recipes_loader(args.english_recipes)
        self.queries, self.contents = chinese_recipes_loader(args.chinese_recipes)
        self.instructions = instruction_loader(args.relevance_instructions)

    def process_queries(self):
        processed_queries = []
        for query, (ingredient, step) in zip(self.queries, self.contents):
            translated_query = translate_query(query, self.args.translated_model)
            generated_query = recipe_query_generation(ingredient, step, self.args.query_processing_LLM_model)
            adapt_query = query_adaptation(query, self.args.rerank_LLM_model)
            processed_queries.append([translated_query, generated_query, adapt_query])
        return processed_queries

    def run(self):
        final_results = []
        processed_queries = self.process_queries()
        for query_set, (query, (ingredient, step)) in zip(processed_queries, zip(self.queries, self.contents)):
            results = get_retrieve_results(query_set, self.args.english_recipes_index, self.english_recipes, self.args.retrieval_model_name, self.args.retrieval_cutoff)
            final_queue = []
            for result in results:
                final_queue.append(rerank(result, query, ingredient, step, self.args.rerank_LLM_model, self.instructions, self.args.max_per_document_length))
            final_result = rerank(final_queue, query, ingredient, step, self.args.rerank_LLM_model, self.instructions, self.args.max_per_document_length)
            final_results.append(final_result)
            print (f"Chinese Recipe: {query}, Final Recipe: {final_result}")
        return final_results
    
    #directly generate result as baseline
    def baseline_generation(self):
        final_results = []
        for query, (ingredient, step) in zip(self.queries, self.contents):
            final_result = generation(query, ingredient, step, self.args.query_processing_LLM_model, self.args.max_per_document_length)
            final_results.append(final_result)
            print (f"Chinese Recipe: {query}, Final Recipe: {final_result}")
        return final_results

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='CARROT Framework Parser')
     parser.add_argument('--chinese_recipes', type=str, default="./ChineseRecipes/test_set.tsv", help='path to chinese recipe file, the text should be title \t ingredients \t steps')
     parser.add_argument('--english_recipes', type=str, default="./EnglishRecipes/english_recipes.tsv", help='English Recieps DataBase tsv, the text should be index \t title \t ingredients \t steps')
     parser.add_argument('--english_recipes_index', type=str, default="./index/all-mpnet-base-v2.index",  help='faiss Documents index path')

     parser.add_argument('--retrieval_model_name', type=str, default='all-mpnet-base-v2', help='retrieval model name')
     parser.add_argument('--retrieval_cutoff', type=int, default=10, help='number of retrieval results per query')
     parser.add_argument('--max_per_document_length', type=int, default=256, help='max_length_each_recipe')

     parser.add_argument('--query_processing_LLM_model', type=str, default="baichuan-inc/Baichuan2-7B-Chat", help='LLM model name for rewriten')
     parser.add_argument('--rerank_LLM_model', type=str, default="baichuan-inc/Baichuan2-7B-Chat", help='LLM model name for rerank')
     parser.add_argument('--translated_model', type=str, default="Helsinki-NLP/opus-mt-zh-en", help='model for translate')

     parser.add_argument('--relevance_instructions', type=str, default="./Instructions/relevance_instruction", help='model for rerank')
     
     parser.add_argument('--baseline_generate', type=str, default=True, help='compare with generated results baseline')
     parser.add_argument('--generation_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='model for generation')
     args = parser.parse_args()

     carrot = Carrot(args)
     #carrot.run()
     if args.baseline_generate:
         carrot.baseline_generation()



          