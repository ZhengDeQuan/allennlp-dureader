#encoding: utf-8

import logging
from log_helper import init_log
init_log('brc')
logger = logging.getLogger('brc')
import pickle
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
#logger = logging.getLogger('global')

class BRCDataLoader(torch.utils.data.DataLoader):#torch.utils.data.DataLoader这个似乎是一个经常被使用的类，有必要细看一哈
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False):
        super(BRCDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, self._collate_fn, pin_memory, drop_last)
    def _collate_fn(self, batch):
        '''
        Args:
            batch: list of dict like
        one_piece = { 'question_token_ids': sample['question_token_ids'],
                      'question_length': len(sample['question_token_ids']),
                      'start_id': sample['answer_spans'][0][0] if 'answer_spans' in sample else None,
                      'end_id': sample['answer_spans'][0][1] if 'answer_spans' in sample else None,
                      'answer_passage_id': sample['answer_passages'][0] if 'answer_passages' in sample else None, 
                      'passage_token_ids': [[],[],[]],
                      'passage_length': [0,0,0]}
            return:

        '''
        batch_size = len(batch)
        max_passage_len = max([max(data['passage_length']) for data in batch])
        max_question_len = max([data['question_length'] for data in batch])
        batch_piece = {'question_token_ids': [],
                      'question_length': [],
                      'start_id': [],
                      'end_id': [],
                      'answer_passage_id': [], 
                      'passage_token_ids': [],
                      'passage_length': [],
                      'sample': []}
        for data in batch:
            for key in {'question_token_ids', 'question_length', 'start_id', 'end_id', 'answer_passage_id'}:
                batch_piece[key].append(data[key])
            assert len(data['passage_token_ids']) == len(data['passage_length'])
            batch_piece['passage_token_ids'] += data['passage_token_ids']
            batch_piece['passage_length'] += data['passage_length']
        max_passage_len =  max(batch_piece['passage_length'])
        max_question_len = max(batch_piece['question_length'])
        batch_piece['passage_token_ids'] = self._pad_sequence(batch_piece['passage_token_ids'], max_passage_len, pad_value = 0)
        batch_piece['question_token_ids'] = self._pad_sequence(batch_piece['question_token_ids'], max_question_len, pad_value = 0)
        for b_idx in range(batch_size):
            answer_passage_id = batch_piece['answer_passage_id'][b_idx]
            start_id = batch_piece['start_id'][b_idx]
            end_id = batch_piece['end_id'][b_idx]
            if answer_passage_id:
                start_id += answer_passage_id * max_passage_len
                end_id += answer_passage_id * max_passage_len
            batch_piece['start_id'][b_idx] = start_id
            batch_piece['end_id'][b_idx] = end_id
        for k,v in batch_piece.items():
            batch_piece[k] = np.array(v)
            #logger.info('key:{} v:{}'.format(k, batch_piece[k].shape))
        batch_piece['sample'] = [data['sample'] for data in batch]
        return batch_piece

    def _pad_sequence(self, sequences, pad_length, pad_value = 0):
        '''
            sequences: list of list of int
            pad_leng: int
            pad_value: int
        '''
        pad_seq = [(ids + [pad_value]*(pad_length - len(ids)))[:pad_length]
                    for ids in sequences]
        return pad_seq

if __name__ == '__main__':
    from .dataset import BRCDataset
    max_passage_num = 5 
    max_passage_len = 500
    max_question_len = 60
    train_files = ['/mnt/lustre/share/wangchangbao/datasets/MRC2018/preprocessed/devset/search.dev.json']
    vocab_dir = '/mnt/lustre/share/wangchangbao/workspace/NaiveReader/search/vocab'
    vocab_dir = '/mnt/lustre/share/wangchangbao/workspace/DuReader/models/combine/vocab'
    
    brc_data = BRCDataset(max_passage_num, max_passage_len, max_question_len, True, train_files)
    print(len(brc_data))
    with open(os.path.join(vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin, encoding='iso-8859-1')
    print('converting ids ....')
    brc_data.convert_to_ids(vocab)

    brc_loader = BRCDataLoader(brc_data, batch_size = 2)
    for i, input in enumerate(brc_loader):
        if i > 5: break
        for k, v in input.items():
            print(k, v.shape)
    
