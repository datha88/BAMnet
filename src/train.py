import timeit
import argparse
#import numpy as np
import torch

from core.bamnet.bamnet import BAMnetAgent
from core.build_data.build_all import build
from core.build_data.utils import vectorize_data
from core.utils.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    cfg = vars(parser.parse_args())
    opt = get_config(cfg['config'])
    print_config(opt)

    # Ensure data is built
    #build(opt['data_dir'])
    #train_vec = load_json(os.path.join(opt['data_dir'], opt['train_data']))
    #valid_vec = load_json(os.path.join(opt['data_dir'], opt['valid_data']))
    #print('GPU Memory at starting: ', torch.cuda.memory_allocated())
    vocab2id = load_json(os.path.join(opt['data_dir'], 'vocab2id.json'))
    ctx_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    '''for indx in range(1000, 1561):
        #indx = 0       
        train_vec = load_json(os.path.join(opt['split_data_dir'], 'train'+str(indx)+'.json'))
        print('train'+str(indx)+'.json')
        
        train_queries, train_raw_queries, train_query_mentions, train_memories, _, train_gold_ans_inds, _ = train_vec
        
        train_queries, train_query_words, train_query_lengths, train_memories = vectorize_data(train_queries, train_query_mentions, \
                                        train_memories, max_query_size=opt['query_size'], \
                                        max_query_markup_size=opt['query_markup_size'], \
                                        max_mem_size=opt['mem_size'], \
                                        max_ans_bow_size=opt['ans_bow_size'], \
                                        max_ans_type_bow_size=opt['ans_type_bow_size'], \
                                        max_ans_path_bow_size=opt['ans_path_bow_size'], \
                                        max_ans_path_size=opt['ans_path_size'], \
                                        fixed_size=True, \
                                        vocab2id=vocab2id)
        train_vec = train_queries, train_raw_queries, train_query_mentions, train_query_words, train_query_lengths, train_memories, train_gold_ans_inds
        dump_json(train_vec, os.path.join(opt['vectorized_split_data_dir'], 'train_'+str(indx)+'.json'))
    
    for indx in range(404):
        valid_vec = load_json(os.path.join(opt['split_data_dir'], 'valid'+str(indx)+'.json'))
        valid_queries, valid_raw_queries, valid_query_mentions, valid_memories, valid_cand_labels, valid_gold_ans_inds, valid_gold_ans_labels = valid_vec
        valid_queries, valid_query_words, valid_query_lengths, valid_memories = vectorize_data(valid_queries, valid_query_mentions, \
                                        valid_memories, max_query_size=opt['query_size'], \
                                        max_query_markup_size=opt['query_markup_size'], \
                                        max_mem_size=opt['mem_size'], \
                                        max_ans_bow_size=opt['ans_bow_size'], \
                                        max_ans_type_bow_size=opt['ans_type_bow_size'], \
                                        max_ans_path_bow_size=opt['ans_path_bow_size'], \
                                        max_ans_path_size=opt['ans_path_size'], \
                                        fixed_size=True, \
                                        vocab2id=vocab2id)
        valid_vec = valid_queries, valid_raw_queries, valid_query_mentions, valid_query_words, valid_query_lengths, valid_memories, valid_cand_labels, valid_gold_ans_inds, valid_gold_ans_labels 
        dump_json(valid_vec, os.path.join(opt['vectorized_split_data_dir'], 'valid_'+str(indx)+'.json'))
        
    '''
    start = timeit.default_timer()
    '''if opt['cuda']:
        print('GPU Memory before init model: ', torch.cuda.memory_allocated())
    '''
    model = BAMnetAgent(opt, ctx_stopwords, vocab2id)
    '''if opt['cuda']:
        print('GPU Memory after init model: ', torch.cuda.memory_allocated())
    '''
    del opt, ctx_stopwords, vocab2id
    '''if opt['cuda']:
        print('GPU Memory before starting train: ', torch.cuda.memory_allocated())
    '''
    model.train()
    
    print('Runtime: %ss' % (timeit.default_timer() - start))
