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