# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import torch
import os
import json
import logging
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import tqdm
import pickle
import logging
logger = logging.getLogger('brc')

#class BRCDataset(torch.utils.data.Dataset):
def display(message):
    logger.info(message)

class BRCDataset(Dataset):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len, is_train,
                 files, rank=0, world_size=1, keep_raw = False):
        self.rank = rank
        self.world_size = world_size
        self.is_train = is_train
        self.keep_raw = keep_raw
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.dataset = []
        for file in files:
            logger.info('loading data from {}'.format(file))
            subset = self._load_dataset(file, train=self.is_train)
            logger.info('data size of {}:{}'.format(file, len(subset)))
            self.dataset += subset
        logger.info('dataset size: {} questions.'.format(len(self.dataset)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        cache_path = os.path.join('cache', data_path.replace('/', '_') + '.{}.{}.pkl'.format(self.rank, self.world_size))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if os.path.exists(cache_path):
            display('load cached from %s' % cache_path)
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
                return dataset
        assert os.path.exists(data_path)
        total, valid, bad = 0, 0, 0
        with open(data_path) as fin:
            data_set = []
            if self.rank == 0:
                fin = tqdm.tqdm(fin)
            for lidx, line in enumerate(fin):
                if self.rank == 0:
                    fin.set_description('processing %d ' % (lidx))
                #divide data
                if lidx % self.world_size != self.rank:
                    continue
                total += 1

                sample = json.loads(line.strip())
                if train:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:
                        continue
                    if len(sample['match_scores']) == 0 or sample['match_scores'][0] < 0.1:
                        continue
                valid += 1#valid这里指的是合法的，不是指验证集

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             #'all_paragraphs_tokens': doc['segmented_paragraphs'],
                             'is_selected': doc['is_selected']}
                        )
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))#recall越大越好，长度越短越好
                        fake_passage_tokens = []
                        for para_info in para_infos:
                            fake_passage_tokens += para_info[0]#按顺序取出para_tokens
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                if not self.keep_raw:
                    del sample['documents']
                    del sample['fake_answers']
                    del sample['question']
                    del sample['segmented_question']
                    del sample['segmented_answers']
                    del sample['match_scores']
                    del sample['fact_or_opinion']
                data_set.append(sample)
        with open(cache_path, 'wb') as cache:
            logger.info('dump cache into %s' % cache_path)
            pickle.dump(data_set, cache)
        logger.info('total:{} valid:{} bad:{}'.format(total, valid, total-valid)) 

        return data_set

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        answer_span = sample.get('answer_spans', [[None, None]])
        if len(answer_span) == 0:
            answer_span = [None, None]
        else:
            answer_span = answer_span[0]
        answer_passages = sample.get('answer_passages', [None])
        if len(answer_passages) == 0:
            answer_passage_id = None
        else:
            answer_passage_id = answer_passages[0]
        if answer_span[0] is not None:
            assert answer_passage_id is not None
        #logger.info('answer_spans:{} ansswer_passage_id:{}'.format(answer_span, answer_passage_id))
        
        one_piece = { 'question_token_ids': sample['question_token_ids'],
                      'question_length': min(self.max_q_len, len(sample['question_token_ids'])),
                      'start_id': answer_span[0],
                      'end_id': answer_span[1],
                      'answer_passage_id': answer_passage_id, 
                      'passage_token_ids': [],
                      'passage_length': [],
                      'sample': sample}
        for passage_idx, passage in enumerate(sample['passages']):
            if passage_idx >= self.max_p_num:
                break
            passage_token_ids = sample['passages'][passage_idx]['passage_token_ids']
            one_piece['passage_token_ids'].append(passage_token_ids)
            one_piece['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
        while len(one_piece['passage_token_ids']) < self.max_p_num:
            one_piece['passage_token_ids'].append([])
            one_piece['passage_length'].append(0)
        assert len(one_piece['passage_token_ids']) == self.max_p_num, '{} vs {}'.format(len(one_piece['passage_token_ids']), self.max_p_num)
        assert len(one_piece['passage_length']) == len(one_piece['passage_token_ids']), '{} vs {}'.format(len(one_piece['passage_length']), len(one_piece['passage_token_ids']))
        #logger.info('get %d' % idx)
        return one_piece


    #deprecated
    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    #deprecated
    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        #if set_name is None:
        #    data_set = self.train_set + self.dev_set + self.test_set
        #elif set_name == 'train':
        #    data_set = self.train_set
        #elif set_name == 'dev':
        #    data_set = self.dev_set
        #elif set_name == 'test':
        #    data_set = self.test_set
        #else:
        #    raise NotImplementedError('No data set named as {}'.format(set_name))
        data_set = self.dataset
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.dataset]:
            #if data_set is None:
            #    continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

