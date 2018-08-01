#coding:utf-8
import sys
import json
import codecs
def reader(input):
    with open(input, 'r') as fin:
        for line in fin:
            yield json.loads(line)
def writer(output):
    return codecs.open(output, 'w', encoding='utf-8')

def add_yesno_answer(input, output):
    output = writer(output)
    for data in reader(input):
        answer = data['answers'][0]
        yesno = 'No' if u'不' in answer else 'Yes'
        print(yesno)
        data['yesno_answers'].append(yesno)
        output.write(json.dumps(data, ensure_ascii=False) + '\n')
    output.close()

def get_segmented_sentences(input, output):
    output = writer(output)
    for d_idx, data in enumerate(reader(input)):
        sys.stdout.write('\b\r processing %d  ' % d_idx)
        for doc in data['documents']:
            for para in doc['segmented_paragraphs']:
                line = ' '.join(para) + '\n'
                output.write(line)
            line = ' '.join(data['segmented_question']) + '\n'
            output.write(line)
    output.close()

if __name__ == '__main__':
    #add_yesno_answer(sys.argv[1], sys.argv[2])
    get_segmented_sentences(sys.argv[1], sys.argv[2])
    print('done')

