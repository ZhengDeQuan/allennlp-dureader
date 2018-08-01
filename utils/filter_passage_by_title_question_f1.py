from preprocess import f1_score, recall
import json
import sys, os

def to_list(text):
    return list(text.replace(' ', ''))
input = sys.argv[1]
output = open(sys.argv[2], 'w')
with open(input, 'r') as fin:
    for qidx, line in enumerate(fin):
        print('processing question: %d ' % qidx)
        anno = json.loads(line)
        question = anno['question']
        documents = anno['documents']
        selected = []
        max_f1, max_doc = 0, None
        for doc in documents:
            title = doc['title'].rsplit('_', 1)[0]
            f1 = recall(to_list(title), to_list(question))
            if f1 > 0.5:
                print('{0} vs {1}:{2}'.format(question, title, f1))
                selected.append(doc)
            if f1 > max_f1:
                max_doc = doc
                max_f1 = f1
        if len(selected) == 0: selected.append(max_doc)
        anno['documents'] = selected
        output.write(json.dumps(anno, ensure_ascii=False) + '\n')
print('done')

