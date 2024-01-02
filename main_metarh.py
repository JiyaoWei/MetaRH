import os
from trainer_metarh import *
from params import *
from data_loader import *
import json
import importlib, sys
importlib.reload(sys)

import itertools, collections
def get_Ground(train_data):
    ground = collections.defaultdict(set)
    for st in train_data:
        ground[tuple(st[:2])].add(st[2])
        if len(st) > 3:
            qualifier = st[3:]
            qualifier_pair = [[qualifier[_*2], qualifier[_*2+1]] for _ in range(len(qualifier)//2)]
            for len_ in range(len(qualifier_pair)):
                for pairs in itertools.combinations(qualifier_pair, len_+1):
                    st_ = []
                    for pair in pairs:
                        st_ += pair
                    ground[tuple(st[:2]+st_)].add(st[2])
    return ground

def tasks_pad(tasks, max_seq_length):
    new_tasks = defaultdict(list)
    for rel in tasks:
        new_sts = []
        for st in tasks[rel]:
            new_sts.append(st + ['[PAD]']*(max_seq_length - len(st)))
        new_tasks[rel] = new_sts
    return new_tasks
def back_pad(back, max_seq_length):
    new_back = []
    for st in back:
        new_back.append(st + ['[PAD]']*(max_seq_length - len(st)))
    return new_back
def Ground_pad(Ground, max_seq_length):
    new_Ground = defaultdict(list)
    for index in Ground:
        new_index = list(index) + ['[PAD]']*(max_seq_length-len(index)-1)
        new_Ground[tuple(new_index)] = Ground[index]
    return new_Ground

if __name__ == '__main__':
    params = get_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = params['device']
    params['device'] = 'cuda:0'

    print("---------Parameters---------")
    for k, v in params.items():
        print(k + ': ' + str(v))
    print("----------------------------")

    # control random seed
    if params['seed'] is not None:
        seed = int(params['seed'])
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        np.random.seed(seed)
        random.seed(seed)
        print('random seed {}'.format(seed))
        torch.use_deterministic_algorithms(True)

    # select the dataset
    for k, v in data_dir.items():
        data_dir[k] = params['data_path']+params['dataset']+v
    

    dataset = dict()
    print("loading train_tasks_in_train ... ...")
    print("loading test_tasks ... ...")
    print("loading dev_tasks ... ...")
    train_tasks = json.load(open(data_dir['train_tasks']))
    background = json.load(open(data_dir['background']))
    test_tasks = json.load(open(data_dir['test_tasks']))
    dev_tasks = json.load(open(data_dir['dev_tasks']))
    dataset['rel2candidates'] = json.load(open(data_dir['rel2candidates_in_train']))

    train_data, test_data, dev_data = [], [], []
    if params['use_in_train']:
        train_data = background

    for rel in train_tasks:
        for st in train_tasks[rel]:
            train_data.append(st)
    for rel in test_tasks:
        for idx, st in enumerate(test_tasks[rel]):
            if idx < params['few']:
                train_data.append(st)
            else:
                test_data.append(st)
    for rel in dev_tasks:
        for idx, st in enumerate(dev_tasks[rel]):
            if idx < params['few']:
                train_data.append(st)
            else:
                dev_data.append(st)
    

    Ground = get_Ground(train_data+test_data+dev_data)
    Ground_t = get_Ground(train_data)

    Ground_t = Ground_pad(Ground_t, params['max_seq_length'])
    dataset['Ground_t'] = Ground_t
    Ground = Ground_pad(Ground, params['max_seq_length'])
    dataset['Ground'] = Ground

    new_train_tasks = defaultdict(list)
    for rel in train_tasks:
        new_train_tasks[rel] = train_tasks[rel]
    if params['use_in_train']:
        for st in background:
            new_train_tasks[st[1]].append(st)

    train_tasks = tasks_pad(new_train_tasks, params['max_seq_length'])
    test_tasks = tasks_pad(test_tasks, params['max_seq_length'])
    dev_tasks = tasks_pad(dev_tasks, params['max_seq_length'])
    background = back_pad(background, params['max_seq_length'])
    if params['use_in_train']:
        all_data = background
    else:
        all_data = []
    for tasks in [train_tasks, test_tasks, dev_tasks]:
        for rel in tasks:
            all_data += tasks[rel]
    for st in all_data:
        index = tuple([st[0], st[1]]+st[3:])
        assert st[2] in Ground[index]

    dataset['train_tasks'] = train_tasks
    dataset['dev_tasks'] = dev_tasks
    dataset['test_tasks'] = test_tasks

    print("----------------------------")

    # data_loader
    train_data_loader = DataLoader(dataset, params, step='train')
    dev_data_loader = DataLoader(dataset, params, step='dev')
    test_data_loader = DataLoader(dataset, params, step='test')
    data_loaders = [train_data_loader, dev_data_loader, test_data_loader]

    # trainer
    trainer = Trainer(data_loaders, dataset, params, background)

    if params['step'] == 'train':
        best_epoch = trainer.train()
        print("test")
        trainer.reload()
        result, bi_result, n_result = trainer.eval(istest=False)
        test_result, test_bi_result, test_n_result = trainer.eval_by_relation(istest=True)
        print('!'*77)
        print("验证集 MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\tBest_epoch: {:.3f}\r".format(
               result['MRR'], result['Hits@10'], result['Hits@5'], result['Hits@3'], result['Hits@1'], best_epoch))
        print("验证集 bi MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\tBest_epoch: {:.3f}\r".format(
               bi_result['MRR'], bi_result['Hits@10'], bi_result['Hits@5'], bi_result['Hits@3'], bi_result['Hits@1'], best_epoch))
        print("验证集 n MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\tBest_epoch: {:.3f}\r".format(
               n_result['MRR'], n_result['Hits@10'], n_result['Hits@5'], n_result['Hits@3'], n_result['Hits@1'], best_epoch))
        print("测试集 MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\tBest_epoch: {:.3f}\r".format(
               test_result['MRR'], test_result['Hits@10'], test_result['Hits@5'], test_result['Hits@3'], test_result['Hits@1'], best_epoch))
        print("测试集 bi MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\tBest_epoch: {:.3f}\r".format(
               test_bi_result['MRR'], test_bi_result['Hits@10'], test_bi_result['Hits@5'], test_bi_result['Hits@3'], test_bi_result['Hits@1'], best_epoch))
        print("测试集 n MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\tBest_epoch: {:.3f}\r".format(
               test_n_result['MRR'], test_n_result['Hits@10'], test_n_result['Hits@5'], test_n_result['Hits@3'], test_n_result['Hits@1'], best_epoch))
    elif params['step'] == 'test':
        trainer.reload()
        test_result, test_bi_result, test_n_result = trainer.eval_by_relation(istest=True)
        print("测试集 MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               test_result['MRR'], test_result['Hits@10'], test_result['Hits@5'], test_result['Hits@3'], test_result['Hits@1']))
    elif params['step'] == 'dev':
        trainer.reload()
        result, test_bi_result, test_n_result = trainer.eval_by_relation(istest=False)
        print("验证集 MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               result['MRR'], result['Hits@10'], result['Hits@5'], result['Hits@3'], result['Hits@1']))
    elif params['step'] == 'case':
        trainer.reload()
        result = trainer.case_eval(istest=True)
        print("验证集 MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               result['MRR'], result['Hits@10'], result['Hits@5'], result['Hits@3'], result['Hits@1']))

