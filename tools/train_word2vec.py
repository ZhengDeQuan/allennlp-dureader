import fasttext
import sys
model = fasttext.skipgram(sys.argv[1], 'word2vec_model')
print('done')
