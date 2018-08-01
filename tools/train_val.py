from __future__ import division
# workaround of the bug where 'import torchvision' sets the start method to be 'fork'
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

import logging
from log_helper import init_log, print_speed
init_log('brc')
init_log('debug', filter_by_rank=False)
logger = logging.getLogger('brc')
debugger = logging.getLogger('debug')

from load_helper import restore_from
from lr_helper import IterLinearLR
from distributed_utils import dist_init, average_gradients, broadcast_params

from utils import compute_bleu_rouge
from utils import normalize

from allennlp.dureader.dataset import BRCDataset
from allennlp.dureader.dataloader import BRCDataLoader
from allennlp.dureader.vocab import Vocab as DuVocab
from allennlp.data import Vocabulary as AllenVocab

from allennlp.common.params import Params
from allennlp.models.reading_comprehension.dureader_bidaf import BidirectionalAttentionFlow as BIDAF

from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from torch.optim.lr_scheduler import MultiStepLR

import codecs
import argparse
import logging
import os
import time

import pickle
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import json
import gensim

parser = argparse.ArgumentParser(description='PyTorch DuReader')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--betas', default=[0.9, 0.9], type=int, nargs='+',
                    help='detas for Adam')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--results_dir', dest='results_dir', default='results_dir',
                    help='results dir of output for each class')
parser.add_argument('--model_dir', dest='model_dir', default='checkpoints',
                    help='directory to save models')
parser.add_argument('--warmup_epochs', dest='warmup_epochs', type=int, default=0,
                    help='epochs for warming up to enlarge lr')
parser.add_argument('--step_epochs', dest='step_epochs', type=lambda x: list(map(int, x.split(','))),
                    default='-1', help='epochs to decay lr')
parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.1, help='rate of lr decay')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--vocab_dir', dest='vocab_dir', required=True,
                    help='directory to store vocab')
parser.add_argument('--word2vec_path', default='',
                    help='path to pretrained word2vec')
parser.add_argument('--train_files', dest='train_files', nargs='+',
                    help = 'json files of trainset')
parser.add_argument('--val_files', dest='val_files', nargs='+',
                    help = 'json file of validate set')
parser.add_argument('--distributed', dest='distributed', action='store_true',
                    help='distributed training or not')
parser.add_argument('--backend', dest='backend', type=str, default='nccl',
                    help='backend for distributed training')
parser.add_argument('--port', dest='port', required=True, help='port of server')
parser.add_argument('--readable', dest='readable', action='store_true',
                    help='save results as human readable')

#hyperparameter
parser.add_argument('--max_p_len', dest='max_p_len', type=int, default=500, help='maximum length of a passage')
parser.add_argument('--max_p_num', dest='max_p_num', type=int, default=5, help='maximum number of passages per question')
parser.add_argument('--max_q_len', dest='max_q_len', type=int, default=60, help='maximum length of a question')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_rank_world_size(args):
    rank, world_size = 0, 1
    if args.distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, world_size
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception as e:
        pass
        #logger.info('error message:{}'.format(e))
    return rank, world_size

def construct_allen_vocab_from_du_vocab(du_vocab):
    logger.info('construct_allen_vocab_from_du_vocab....')
    allen_vocab = AllenVocab()
    tokens = [du_vocab.get_token(idx) for idx in range(du_vocab.size())]
    for token in tokens:
        allen_vocab.add_token_to_namespace(token)
    for token in tokens:
        du_idx = du_vocab.get_id(token)
        allen_idx = allen_vocab.get_token_index(token)
        assert du_idx == allen_idx, 'token:{} du:{} vs allen:{}'.format(token, du_idx, allen_idx)
    a = du_vocab.size()
    b = allen_vocab.get_vocab_size()
    assert a == b , 'du_vocab size:{} vs allen_vocab size:{}'.format(a,b)
    logger.info('Allen Vocabulary prepared!')
    return allen_vocab

def prepare_du_vocab(vocab_dir, trainset, embed_size, args):
    vocab_file = os.path.join(vocab_dir, 'vocab.data')
    if os.path.exists(vocab_file):
        logger.info('loading vocabulary from %s ' % vocab_file)
        with open(vocab_file, 'rb') as fin:
            vocab = pickle.load(fin)
            return vocab
    else:
        vocab = DuVocab(lower=False)
        if args.word2vec_path != '':
            word2vec = gensim.models.Word2Vec.load(args.word2vec_path)
            logger.info('loading voabulary from %s ' % args.word2vec_path)
            for word in word2vec.wv.vocab:
                vocab.add(word)
        else:
            for word in trainset.word_iter():
                vocab.add(word)
            unfiltered_vocab_size = vocab.size()
            vocab.filter_tokens_by_cnt(min_cnt=2)
            filtered_num = unfiltered_vocab_size - vocab.size()
            logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                                    vocab.size()))
        logger.info('vocab size:{}'.format(vocab.size()))
        logger.info('Assigning embeddings...')
        vocab.randomly_init_embeddings(embed_size)
        logger.info('Saving vocab...')
        with open(vocab_file, 'wb') as fout:
            pickle.dump(vocab, fout)
        logger.info('Done with preparing!')
        return vocab
    logger.critical('failed to prepare vocabulary')

def build_dataloader(files, args, is_train=True):
    rank, world_size = get_rank_world_size(args)
    dataset = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, 
            is_train=is_train, files=files, rank=rank, world_size=world_size, keep_raw = not is_train)
    dataloader = BRCDataLoader(dataset, 
            batch_size=args.batch_size, 
            shuffle=True if is_train else False,
            sampler=None, batch_sampler=None,
            num_workers=args.workers)
    return dataloader

def warmup(model, train_loader, optimizer, args):
    rank, world_size = get_rank_world_size(args)
    target_lr = args.lr * max(world_size,1)
    warmup_iter = args.warmup_epochs * len(train_loader)
    gamma = (target_lr - args.lr)/ warmup_iter
    lr_scheduler = IterLinearLR(optimizer, gamma)
    for epoch in range(args.warmup_epochs):
        logger.info('warmup epoch %d' % (epoch))
        train(train_loader, model, lr_scheduler, epoch, args, warmup=True)
    # overwrite initial_lr with magnified lr through warmup
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    logger.info('warmup for %d epochs done, start large batch training' % args.warmup_epochs)


def makedirs(dirs):
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def main():
    args = parser.parse_args()
    params = Params.from_file(args.config)#args.config==bidaf.json
    args.embed_size = params.get('model').get('text_field_embedder').get('tokens').get('embedding_dim')
    logger.info('argg.embed_size:{}'.format(args.embed_size))
    makedirs([args.vocab_dir, args.results_dir, args.model_dir])
    if args.distributed:
        logger.info('initialize for distributed training, port:{}, backend:{}'.format(args.port, args.backend))
        dist_init(args.port, backend = args.backend)
    rank, world_size = get_rank_world_size(args)
    
    # build dataset
    train_loader = build_dataloader(args.train_files, args, is_train=True)
    val_loader = build_dataloader(args.val_files, args, is_train=False)
    du_vocab = prepare_du_vocab(args.vocab_dir, train_loader.dataset, args.embed_size, args)
    allen_vocab = construct_allen_vocab_from_du_vocab(du_vocab)
    train_loader.dataset.convert_to_ids(du_vocab)
    val_loader.dataset.convert_to_ids(du_vocab)
    logger.info('build dataloader done')
    
    # build model
    model = BIDAF.from_params(vocab=allen_vocab, params=params.pop('model'))
    logger.info('build model done:{}'.format(model))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params,
                                lr=args.lr,
                                betas=args.betas,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch = restore_from(model, optimizer, args.resume)

    model = model.cuda()
    if args.distributed:
        broadcast_params(model)

    if args.evaluate:
        validate(val_loader, model, args)
        return

    # warmup to enlarge lr
    if args.start_epoch == 0 and args.warmup_epochs > 0:
        warmup(model, train_loader, optimizer, args)
    
    lr_scheduler = MultiStepLR(optimizer, milestones=args.step_epochs, gamma=args.decay_rate, last_epoch=args.start_epoch-1)
    #reduce_on_plateau = False
    #if params.get('learning_rate_scheduler', None) is not None:
    #    lr_scheduler = LearningRateScheduler.from_params(params.get('learning_rate_scheduler'))
    #    reduce_on_plateau = True
    #    logger.info('using reduce_on_plateau scheduler:{}'.format(lr_scheduler.__class__.__name__))

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        # train for one epoch
        train(train_loader, model, lr_scheduler, epoch, args)
        val_loss = validate(val_loader, model, args)
        #if reduce_on_plateau:
        #    lr_scheduler.step(val_loss)
        #lr = lr_scheduler.get_lr()[0]

        if rank == 0:
            os.makedirs(args.model_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.cpu().state_dict(),
                'optimizer': optimizer.state_dict()}
            save_path = os.path.join(args.model_dir, 'checkpoint_e%d.pth' % (epoch + 1))
            torch.save(checkpoint, save_path)

def train(train_loader, model, lr_scheduler, epoch, args, warmup=False):
    model.cuda()
    model.train()
    rank, world_size = get_rank_world_size(args)
    tik = time.time()
    #logger.info('mark')
    smooth_loss = AverageMeter()
    speed = AverageMeter()
    for iter, input in enumerate(train_loader):
        if warmup:
            # update lr for each iteration
            lr_scheduler.step()
        batch_size = len(input['sample'])
        fake_batch_size = input['passage_token_ids'].shape[0]
        assert fake_batch_size % batch_size == 0, 'fake:{} batch_size:{}'.format(fake_batch_size, batch_size)
        # shape [batch_size * max_p_num, max_p_len]
        passage_token_ids = Variable(torch.from_numpy(input['passage_token_ids']).cuda().long())
        # shape [batch_size, max_long]
        question_token_ids = Variable(torch.from_numpy(input['question_token_ids']).cuda().long())
        span_start = Variable(torch.from_numpy(input['start_id']).cuda().long())
        span_end = Variable(torch.from_numpy(input['end_id']).cuda().long())
        #debugger.info('input shape:{} {}'.format(passage_token_ids.shape, question_token_ids.shape))
        passage = {'tokens': passage_token_ids}
        question = {'tokens': question_token_ids}
        samples = input['sample']
        # shape [batch_sizie, max_p_num * max_p_len]
        output = model(question, passage, span_start, span_end)
        loss = output['loss'] / world_size
        loss_value = loss.cpu().data[0] * world_size
        model.zero_grad()
        loss.backward()
        if args.distributed:
            average_gradients(model)
        lr_scheduler.optimizer.step()
        tok = time.time()
        lr = lr_scheduler.get_lr()[0]
        smooth_loss.update(loss_value)
        if iter < 5:
            this_speed = tok-tik
        else:
            speed.update(tok-tik)
            this_speed = speed.avg
        logger.info('Epoch: [%d][%d/%d] LR:%f Time: %.3f/%.3f Loss: %.5f/%.5f' %
              (epoch, iter, len(train_loader), lr, tok - tik, speed.avg, loss_value, smooth_loss.avg))
        print_speed((epoch) * len(train_loader) + iter + 1, this_speed, args.epochs * len(train_loader))
        tik = tok

def validate(val_loader, model, args):
    model.cuda()
    model.eval()
    rank, world_size = get_rank_world_size(args)
    logger.info('start validate')
    os.makedirs(args.results_dir, exist_ok=True)
    pred_answers, ref_answers = [], []
    tik = time.time()
    val_loss = AverageMeter()
    for iter, input in enumerate(val_loader):
        # shape [batch_size * max_p_num, max_p_len]
        passage_token_ids = Variable(torch.from_numpy(input['passage_token_ids']).cuda().long())
        # shape [batch_size, max_q_len]
        question_token_ids = Variable(torch.from_numpy(input['question_token_ids']).cuda().long())
        passage = {'tokens': passage_token_ids}
        question = {'tokens': question_token_ids}
        #has_ans_span = input['start_id'][0] is not None
        #span_start = None if has_ans_span else Variable(torch.from_numpy(input['start_id']).cuda().long())
        #span_end = None if has_ans_span else Variable(torch.from_numpy(input['end_id']).cuda().long())
        output = model(question, passage)
        #if 'loss' in output:
        #    loss = output['loss']
        #    torch.distributed.all_reduce(loss)
        #    val_loss.update(loss.cpu().data[0])
        tok = time.time()
        best_spans = output['best_span']
        best_scores = output['best_score']
        samples = input['sample']
        batch_size = len(samples)
        batch_pred_answers = get_pred_answers(best_spans, best_scores, samples)
        batch_ref_answers = get_ref_answers(samples)
        pred_answers += batch_pred_answers
        ref_answers += batch_ref_answers
        a = len(batch_pred_answers)
        b = len(batch_ref_answers)
        #assert a == b or a == 0, 'len(batch_pred_answers):{} vs len(batch_ref_answers):{}'.format(a, b)
        logger.info('Test: [%d/%d] Time: %.3f'%(iter, len(val_loader), tok-tik))
        print_speed(iter + 1, tok - tik, len(val_loader))
        tik = tok
    if len(ref_answers) > 0:
        bleu_rouge = evaluate(pred_answers, ref_answers)
        logger.info('bleu_rouge:{}'.format(bleu_rouge))
    dump_pred_results(pred_answers, os.path.join(args.results_dir, 'results.txt.rank%d' % rank), readable=args.readable)
    return val_loss.avg
    
def dump_pred_results(pred_answers, output_file, readable = None):
    logger.info('saving results to {}'.format(output_file))
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with codecs.open(output_file, 'w', encoding='utf-8') as fout:
        for answer in pred_answers:
            if 'sample' in answer and not readable:
                del answer['sample']
            fout.write(json.dumps(answer, ensure_ascii=False, indent=2 if readable else None) + '\n')

def evaluate(pred_answers, ref_answers):
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None
    return bleu_rouge

def get_ref_answers(samples):
    ref_answers = []
    for sample in samples:
        if 'answers' in sample:
            ref_answers.append({'question_id': sample['question_id'],
                         'question_type': sample['question_type'],
                         'answers': sample['answers'],
                         'entity_answers': [[]],
                         'yesno_answers': []})
    return ref_answers

def get_pred_answers(best_spans, best_scores, samples):
    pred_answers = []
    batch_size = len(samples) 
    fake_batch_size = best_spans.shape[0]
    max_p_num = fake_batch_size // batch_size
    title_scores = np.zeros(best_spans.shape[0], dtype=np.float32)
    for b_ix, sample in enumerate(samples):
        q_str = sample['question']
        ref = {0:normalize([q_str])}
        scores = []
        for p_ix in range(max_p_num):
            if p_ix >= len(sample['documents']):
                scores.append(0)
            else:
                doc = sample['documents'][p_ix]
                title = doc['title']
                pred = {0:normalize([title])}
                score = compute_bleu_rouge(pred, ref)
                scores.append(score['Rouge-L'])
        # softmax
        scores = np.exp(scores)/np.sum(np.exp(scores), axis=0)
        title_scores[b_ix*max_p_num:b_ix*max_p_num+max_p_num] = scores
    best_scores = best_scores * title_scores

    answer_spans, answer_scores, answer_docs = find_best_answer(best_spans, best_scores, batch_size)
    for b_idx in range(batch_size):
        sample = samples[b_idx]
        answer_str = get_answer_str(answer_docs[b_idx], answer_spans[b_idx], sample)
        pred_answers.append({'question_id': sample['question_id'],
                 'question_type': sample['question_type'],
                 'answers': [answer_str],
                 'entity_answers': [[]],
                 'yesno_answers': ['Yes'],
                 'sample': sample})
    return pred_answers


def find_best_answer(best_spans, best_scores, batch_size):
    '''
    Args:
        best_spans: [batch_size * max_p_num, 2]
        best_scores: [batch_size * max_p_num]
        batch_size: batch_size
    returns:
        answer_spans: [batch_size, 2]
        answer_scores: [batch_size]
        answer_doc_id: [batch_size]
    '''
    fake_batch_size = best_spans.shape[0]
    max_p_num = fake_batch_size // batch_size
    assert fake_batch_size % batch_size == 0, 'fake:{} batch_size:{}'.format(fake_batch_size, batch_size)
    answers = {'answer_spans': [], 'answer_scores':[], 'answer_docs':[]}
    answer_spans, answer_scores, answer_docs = [], [], []
    for b_idx in range(batch_size):
        start_p = b_idx * max_p_num
        end_p = start_p + max_p_num
        spans = best_spans[start_p:end_p, :]
        scores = best_scores[start_p:end_p]
        order = np.argsort(scores)[::-1]
        answer_spans.append(spans[order[0]])
        answer_scores.append(scores[order[0]])
        answer_docs.append(order[0])
    return np.array(answer_spans), np.array(answer_scores), np.array(answer_docs)

def get_answer_str(answer_doc, answer_span, sample):
    if len(sample['passages']) == 0:
        print('empty passages of question:{}'.format(sample['question_id']))
        return ''
    p = sample['passages']
    a = answer_doc
    b = len(p)
    assert a < b, 'answer_doc vs num-of-passage:{} vs {}, qid:{}'.format(a, b, sample['question_id'])
    t = p[a]['passage_tokens']
    a = answer_span[0]
    b = answer_span[1]
    c = len(t)
    assert 0 <= a and a <= b and b <= c, 'length of passage vs span[0] vs span[1], qid:{}'.format(c, a, b, sample['question_id'])
    answer = ''.join(sample['passages'][answer_doc]['passage_tokens'][answer_span[0]:answer_span[1]+1])
    return answer

if __name__ == '__main__':
    main()
