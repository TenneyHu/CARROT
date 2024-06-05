import argparse
from data_loader import ir_test_set_loader
import numpy as np
from collections import defaultdict
import os
from prettytable import PrettyTable

class CARROT_IR_Eval:
    def __init__(self, args):
        self.args = args
        self.annotation, self.query_res = ir_test_set_loader(args.test_set)
        self.english_recipes_folder = args.english_recipes_folder

    def load_dataset(self):
        if self.args.switch_eval_on_condensed:
            return self.labeled_dataset
        else:
            return self.dataset 
    def dcg(self, x):
        dcg = 0.0
        for i in range(len(x)):
            dcg += (2 ** x[i] - 1) / np.log2(i + 2)
        return dcg
    
    def ndcg(self):
        NDCG = 0
        dataset = self.load_dataset()

        for qid in dataset:
            if self.query_res[qid] == []:
                continue
            true_relevance = []
            labeled_res = sorted(self.query_res[qid], reverse = True)
            idcg = self.dcg(labeled_res)

            for _, docid  in enumerate(dataset[qid]):
                id = qid + " " + docid
                if id in self.annotation:
                    label = self.annotation[id]
                else:
                    label = 0
                true_relevance.append(label)
            dcg = self.dcg(true_relevance)

            NDCG += dcg / idcg

        res =  NDCG / len(self.query_res)
        return f"{res:.3f}"

    def recall(self, theresold):
        R = 0.0
        samples = 0
        dataset = self.load_dataset()

        for qid in dataset:
            if self.query_res[qid] == []:
                continue
            count = 0
            all_count = 0
            for label in self.query_res[qid]:
                if label >= theresold:
                    all_count += 1
            if all_count > 0:
                samples += 1

            for docid  in self.dataset[qid]:
                id = qid + " " + docid
                if id in self.annotation:
                    label = self.annotation[id]
                else:
                    label = 0
                if label >= theresold:
                    count += 1
            if all_count > 0:
                R += 100.0 * count / all_count
        res = R / samples
        return f"{res:.2f}"
    
    def precision(self, theresold, max_position = 10):
        P = 0.0
        
        dataset = self.load_dataset()
        for qid in dataset:
            if self.query_res[qid] == []:
                continue
            relevant_documents_count = 0.0
            for i, docid in enumerate(self.dataset[qid]):
                id = qid + " " + docid
                if id in self.annotation:
                    label = self.annotation[id]
                else:
                    label = 0

                if label >= theresold:
                    relevant_documents_count += 1

                if i == max_position - 1:
                    break
            P += 100.0 * relevant_documents_count / max_position
        P /= len(self.query_res)
        return f"{P:.2f}"
    
    def mAP(self, theresold):
        mAP = 0.0
        dataset = self.load_dataset()
        for qid in dataset:
            relevant_documents_count = 0.0
            sum_p = 0.0
            if self.query_res[qid] == []:
                continue
            for i, docid in enumerate(self.dataset[qid]):
                id = qid + " " + docid
                if id in self.annotation:
                    label = self.annotation[id]
                else:
                    label = 0

                if label >= theresold:
                    relevant_documents_count += 1
                    sum_p += 1.0 * relevant_documents_count / (i + 1)

            if relevant_documents_count > 0: 
                mAP += 100.0 * sum_p / relevant_documents_count

        res = mAP / len(self.query_res)
        return f"{res:.2f}"
    
    #evaluate all files under the folder
    def run_under_folder(self):
        table = PrettyTable()
        table.field_names = ["Method", "NDCG", "Recall_exact@10", "MAP_exact@10",  
                             "Precision_exact@10", "Precision_exact@1",  "Annotation_count"]

        for filename in os.listdir(self.english_recipes_folder):
            method = filename.split(".")[0]
            annotation_count = 0

            with open(os.path.join(self.english_recipes_folder, filename), 'r') as f:

                self.dataset = defaultdict(list)
                self.labeled_dataset = defaultdict(list)
                for line in f:
                    line  = line.strip().split('\t')
                    qid = line[1]  
                    docid = line[2]  
                    id = qid + " " + docid
                    if id in self.annotation:
                        annotation_count +=1
                        if len(self.labeled_dataset[qid]) <= self.args.pos_cutoff: 
                            self.labeled_dataset[qid].append(docid)
                    if len(self.dataset[qid]) <= self.args.pos_cutoff:        
                        self.dataset[qid].append(docid)

                ndcg_score = self.ndcg()
                recall_exact_match = self.recall(2)

                map_exact_match = self.mAP(2)

                precision_exact_match = self.precision(2, 10)

                precision1_exact_match = self.precision(2, 1)

                table.add_row([method, ndcg_score, recall_exact_match, map_exact_match, 
                               precision_exact_match, precision1_exact_match, annotation_count])
                
        print(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARROT IR evaluate')
    
    parser.add_argument('--test_set', type=str, default="./IR_Dataset/test_set.tsv", help="human annanotion test set")
    parser.add_argument('--english_recipes_folder', type=str, default="./IR_Dataset/results", help='predict results dictionary')
    parser.add_argument('--pos_cutoff', type=int, default=10, help='position cutoff')
    parser.add_argument('--switch_eval_on_condensed', type=bool, default=True, help='if the switch is on, non labelled samples are discarded)')
    args = parser.parse_args()

    carrot_ir = CARROT_IR_Eval(args)
    carrot_ir.run_under_folder()