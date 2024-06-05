from collections import defaultdict
def chinese_recipes_loader(filepath):
    query = []
    content = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            query.append(line[0])
            content.append((line[1],line[2]))
    return query, content

def english_recipes_loader(filepath):
    recipes = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            try:
                id = int(line[0])
                title = line[1]
                ingredients = line[2]
                steps = line[3]
                content = "Title: " + title + ", Ingredients: " + ingredients + ", Steps: " + steps
                recipes[id] = content
            except:
                pass
    return recipes
            
def instruction_loader(filepath):
    instructions = ""
    with open(filepath, "r") as f:
        instructions = f.read().replace("\n", " ")
    return instructions

def reference_loader(filepath):
    references = []
    with open(filepath, "r") as f:
        for line in f:
            references.append([line.strip()])
    return references

def result_loader(filepath):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            results.append(line.strip())
    return results

def ir_test_set_loader(filepath):
    ir_test_set = {}
    ir_test_set_query = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            ir_test_set[line[1]+ " "+ line[2]] = int(line[0])
            ir_test_set_query[line[1]].append(int(line[0]))

    return ir_test_set, ir_test_set_query