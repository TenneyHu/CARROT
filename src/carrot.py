import argparse
from tqdm import tqdm
from query_processing import translate_query, recipe_query_generation, query_adaptation
from data_loader import chinese_recipes_loader, english_recipes_loader, instruction_loader
from retrieve import get_retrieve_results
from rerank import rerank

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='CARROT Framework Parser')
     parser.add_argument('--input', type=str, default="./ChineseRecipes/examples.tsv", help='path to chinese recipe file, the text should be title \t ingredients \t steps')
     parser.add_argument('--output', type=str, default=None, help='path to output file')
     parser.add_argument('--index', type=str, default="./index/all-mpnet-base-v2.index",  help='faiss Documents index path')
     parser.add_argument('--document', type=str, default="./EnglishRecipes/english_recipes.tsv", help='English Recieps DataBase tsv, the text should be index \t title \t ingredients \t steps')

     parser.add_argument('--retrieval_model_name', type=str, default='all-mpnet-base-v2', help='retrieval model name')
     parser.add_argument('--retrieval_model_dimension', type=int, default=768, help='dimension of retrieval model')
     parser.add_argument('--retrieval_cutoff', type=int, default=10, help='number of retrieval results per query')
     parser.add_argument('--max_per_document_length', type=int, default=300, help='max_length_each_recipe')

     parser.add_argument('--query_processing_LLM_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='LLM model name for rewriten')
     parser.add_argument('--rerank_LLM_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='LLM model name for rerank')
     parser.add_argument('--translated_model', type=str, default="Helsinki-NLP/opus-mt-zh-en", help='model for translate')

     parser.add_argument('--relevance_instructions', type=str, default="./Instructions/relevance_instruction", help='model for rerank')
     parser.add_argument('--compare_baseline', action='store_true', default=True, help='output the baseline results')
     args = parser.parse_args()

     english_recipes = english_recipes_loader(args.document)
     queries, contents = chinese_recipes_loader(args.input)
     instructions = instruction_loader(args.relevance_instructions)

     for (query , (ingredient, step)) in zip(queries, contents):
          #Query processing
          translated_query = translate_query(query, args.translated_model)
          generated_query = recipe_query_generation(ingredient, step, args.query_processing_LLM_model)
          adapt_query = query_adaptation(query, args.rerank_LLM_model)
          querys = [translated_query, generated_query, adapt_query]
          
          #retrieval
          print (querys)
          results = get_retrieve_results(querys, args.index, english_recipes, args.retrieval_model_name, args.retrieval_cutoff)
          if args.compare_baseline:
               print ("Baseline Result:", results[0][0])
          
          #Rerank
          final_queue = []
          for result in results:
               final_queue.append(rerank(result, query, ingredient, step, args.rerank_LLM_model, instructions, args.max_per_document_length))
          
          final_result = rerank(final_queue, query, ingredient, step, args.rerank_LLM_model, instructions, args.max_per_document_length)     
          print (final_result)