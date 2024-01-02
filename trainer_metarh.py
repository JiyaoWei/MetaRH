from models_metarh import *
import os
import torch
import random
from collections import defaultdict
from torch.autograd import Variable
import numpy as np
import wandb

class Trainer:
    def __init__(self, data_loaders, dataset, parameter, background):
        self.parameter = parameter
        # data loader
        self.background = background
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.max_seq_length = parameter['max_seq_length']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        # device
        self.device = parameter['device']
        self.embed_dim = parameter['embed_dim']

        self.data_path = parameter['data_path']

        self.embed_model = parameter['embed_model']
        self.max_neighbor = parameter['max_neighbor']
        self.weight = parameter['weight']
        self.dropout_g = parameter['dropout_g']
        self.dropout_i = parameter['dropout_i']
        self.num_layers = parameter['num_layers']
        self.num_heads = parameter['num_heads']
        self.use_neighbors = parameter['use_neighbors']
        self.use_pretrain = parameter['use_pretrain']
        self.fine_tune = parameter['fine_tune']
        self.weight_decay = parameter['weight_decay']

        self.shuffle_background = parameter['shuffle_background']
        self.model_name = self.parameter['dataset']+'_'+str(self.parameter['use_pretrain'])+'_'+str(self.learning_rate)+'_'+str(self.weight)+'_'+str(parameter['embed_dim'])+'_'+str(self.max_neighbor)+'_'+self.embed_model+'_'+str(self.batch_size)+'_'+str(self.few)+'_'+str(self.dropout_i)+'_'+str(self.dropout_g)+'_'+str(self.num_heads)+'_'+str(self.num_layers)+'_'+str(self.max_seq_length)+'_'+self.parameter['relation_learner']+'_'+self.parameter['embedding_learner']+'_'+self.parameter['few_rel_aggregate']+'_'+str(parameter['eval_epoch'])+'_'+str(self.shuffle_background)+'_'+str(parameter['use_neighbors'])+'_'+str(parameter['ablation'])+'_'+str(parameter['fine_tune'])+'_'+str(parameter['weight_decay'])+'_'+str(parameter['use_in_train'])
        
        wandb.init(project='Ours_'+self.parameter['dataset'][2:], name=self.model_name)
        wandb.config.dataset = parameter['dataset']
        wandb.config.use_in_train = parameter['use_in_train']
        wandb.config.weight_decay = parameter['weight_decay']
        wandb.config.use_pretrain = parameter['use_pretrain']
        wandb.config.few_rel_aggregate = parameter['few_rel_aggregate']
        wandb.config.embed_dim = parameter['embed_dim']
        wandb.config.embedding_learner = parameter['embedding_learner']
        wandb.config.shuffle_background = parameter['shuffle_background']
        wandb.config.relation_learner = parameter['relation_learner']
        wandb.config.weight = self.weight
        wandb.config.dropout_g = self.dropout_g
        wandb.config.dropout_i = self.dropout_i
        wandb.config.num_heads = self.num_heads
        wandb.config.num_layers = self.num_layers
        wandb.config.max_seq_length = self.max_seq_length
        wandb.config.max_neighbor = self.max_neighbor
        wandb.config.few = self.few
        wandb.config.learning_rate = self.learning_rate
        wandb.config.embed_model = self.embed_model
        wandb.config.fine_tune = self.fine_tune
        wandb.config.use_neighbors = self.use_neighbors
        wandb.config.use_pretrain = self.use_pretrain
        wandb.config.fine_tune = self.fine_tune
        wandb.config.batch_size = self.batch_size
        self.wandb = wandb

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.ent2id = dict()
        with open("Dataset/%s/ents.txt"%(parameter['dataset'])) as f:
            for line in f.readlines():
                line = line.strip()
                self.ent2id[line] = len(self.ent2id)
        self.num_ents = len(self.ent2id.keys())
        self.build_connection(max_=self.max_neighbor)
        self.id2ent = {id:ent for ent, id in self.ent2id.items()}
        dataset['rel2id'] = self.rel2id
        dataset['ent2id'] = self.ent2id
        dataset['ent2emb'] = self.ent2emb
        dataset['rel2emb'] = self.rel2emb
        self.flen = Flen(dataset, parameter, embed = self.symbol2vec)
        self.flen.to(self.device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.flen.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)

        edge_labels = []
        max_aux = (self.max_seq_length - 3)//2
        edge_labels.append([0, 1, 2] + [3, 0] * max_aux)
        edge_labels.append([1] + [0] * (self.max_seq_length - 1))
        edge_labels.append([2] + [0] * (self.max_seq_length - 1))
        for idx in range(max_aux):
            edge_labels.append(
                [3, 0, 0] + [0, 0] * idx + [0, 4] + [0, 0] * (max_aux - idx - 1))
            edge_labels.append(
                [0, 0, 0] + [0, 0] * idx + [4, 0] + [0, 0] * (max_aux - idx - 1))
        self.edge_labels = np.asarray(edge_labels).astype("int64")


    def load_embed(self):
        # gen symbol2id, without embedding

        symbol_id = {}
        rel2id, ent2id = dict(), dict()

        with open("Dataset/%s/rels.txt"%(self.parameter['dataset'])) as f:
            for line in f.readlines():
                line = line.strip()
                rel2id[line] = len(rel2id)
        with open("Dataset/%s/ents.txt"%(self.parameter['dataset'])) as f:
            for line in f.readlines():
                line = line.strip()
                ent2id[line] = len(ent2id)

        print('LOADING PRE-TRAINED EMBEDDING')
        if self.use_pretrain:
            ent_embed = np.loadtxt(os.path.join("Dataset/%s/entities2vec.%s"%(self.parameter['dataset'], self.parameter['embed_model'])))
            rel_embed = np.loadtxt(os.path.join("Dataset/%s/relations2vec.%s"%(self.parameter['dataset'], self.parameter['embed_model'])))  # contain inverse edge
        else:
            ent_embed = np.random.random((len(ent2id),self.embed_dim))
            rel_embed = np.random.random((len(rel2id),self.embed_dim))  # contain inverse edge

        assert ent_embed.shape[0] == len(ent2id.keys())
        assert rel_embed.shape[0] == len(rel2id.keys())
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.ent2emb = ent_embed
        self.rel2emb = rel_embed
        i = 0
        embeddings = []
        for key in rel2id.keys():
            if key == '[PAD]' or key == '[MASK]':
                continue
            symbol_id[key] = i
            i += 1
            embeddings.append(list(rel_embed[rel2id[key], :]))

        for key in ent2id.keys():
            if key == '[PAD]' or key == '[MASK]':
                continue
            symbol_id[key] = i
            i += 1
            embeddings.append(list(ent_embed[ent2id[key], :]))

        symbol_id['[PAD]'] = i
        embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        embeddings = np.array(embeddings)
        assert embeddings.shape[0] == len(symbol_id.keys())

        self.symbol2id = symbol_id
        self.symbol2vec = embeddings
        assert self.rel2id['[PAD]'] == self.ent2id['[PAD]'] == 0
        self.pad_id = 0

    def build_connection(self, max_=100):
        self.e1_rele2 = defaultdict(list)
        # background是symbol_id
        for st in self.background:
            st_ids = [self.ent2id[st[ind]] if ind%2==0 else self.rel2id[st[ind]] for ind in range(len(st))]
            self.e1_rele2[st[0]].append(tuple(st_ids[1:]))

        if not self.shuffle_background:
            self.connections = (np.ones((self.num_ents, max_, self.parameter['max_seq_length']-1)) * self.pad_id).astype(int)

            for ent, id_ in self.ent2id.items():
                neighbors = self.e1_rele2[ent]
                if len(neighbors) > max_:
                    neighbors = neighbors[:max_]
                for idx, _ in enumerate(neighbors):
                    for ind in range(self.max_seq_length-1):
                        self.connections[id_, idx, ind] = _[ind]

    def get_meta(self, left, right):
        # 1024
        if not self.shuffle_background:
            left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).to(self.device)
            right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).to(self.device)
        else:
            left_connections = np.full((len(left), self.max_neighbor, self.max_seq_length-1), self.pad_id)
            right_connections = np.full((len(right), self.max_neighbor, self.max_seq_length-1), self.pad_id)
            for ind_e, ent in enumerate(left):
                cur_ent = self.id2ent[ent]
                for ind_n, neighbor in enumerate(self.e1_rele2[cur_ent][:self.max_neighbor]):
                    left_connections[ind_e][ind_n] = neighbor
            for ind_e, ent in enumerate(right):
                cur_ent = self.id2ent[ent]
                for ind_n, neighbor in enumerate(self.e1_rele2[cur_ent][:self.max_neighbor]):
                    right_connections[ind_e][ind_n] = neighbor
            left_connections = Variable(torch.LongTensor(left_connections)).to(self.device)
            right_connections = Variable(torch.LongTensor(right_connections)).to(self.device)

        return (left_connections, right_connections)

    def reload(self):
        state_dict_file = os.path.join(self.state_dir, self.model_name + '.ckpt')
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.flen.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self):
        torch.save(self.flen.state_dict(), os.path.join(self.state_dir, self.model_name + '.ckpt'))
            
    def rank_predict(self, data, x, ranks, bi_data, n_data, len_x):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        pre_ent_ind = list(idx.cpu().numpy())[0]
        ranks.append(rank)

        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank <= 3:
            data['Hits@3'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank
        if len_x >3:
            if rank <= 10:
                n_data['Hits@10'] += 1
            if rank <= 5:
                n_data['Hits@5'] += 1
            if rank <= 3:
                n_data['Hits@3'] += 1
            if rank == 1:
                n_data['Hits@1'] += 1
            n_data['MRR'] += 1.0 / rank
        else:
            if rank <= 10:
                bi_data['Hits@10'] += 1
            if rank <= 5:
                bi_data['Hits@5'] += 1
            if rank <= 3:
                bi_data['Hits@3'] += 1
            if rank == 1:
                bi_data['Hits@1'] += 1
            bi_data['MRR'] += 1.0 / rank
        return rank, pre_ent_ind

    def do_one_step(self, task, iseval=False, curr_rel='', istest=False):
        loss, p_score, n_score = 0, 0, 0
        support = task[0]
        support_left = [self.ent2id[few[0]] for batch in support for few in batch]
        support_right = [self.ent2id[few[2]] for batch in support for few in batch]
        edge_labels = torch.tensor(self.edge_labels, dtype=torch.long, device=self.device)
        if iseval == False:
            meta_left = [[0]*self.batch_size for i in range(self.few)]
            meta_right = [[0]*self.batch_size for i in range(self.few)]
        if iseval == True:
            meta_left = [[0] for i in range(self.few)]
            meta_right = [[0] for i in range(self.few)]

        for i in range(len(meta_left)):
            for j in range(len(meta_left[0])):
                meta_left[i][j] = support_left[j*self.few + i]
        for i in range(len(meta_right)):
            for j in range(len(meta_right[0])):
                meta_right[i][j] = support_right[j*self.few + i]
            
        support_meta = []
        # 5*1024
        for i in range(len(meta_left)):
            support_meta.append(self.get_meta(meta_left[i], meta_right[i]))
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.flen(task, iseval, curr_rel, support_meta, istest, edge_labels)
            y = torch.Tensor([[1]]).to(self.device).expand([p_score.shape[0], p_score.shape[1]])
            loss = self.flen.loss_func(p_score, n_score, y)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            n_score = torch.tensor([]).to(self.device)
            for _ in range(0, len(task[3][0]), self.batch_size):
                if _!=0:
                    with torch.no_grad():
                        task_ = [task[0], task[1], task[2], [task[3][0][_:_+self.batch_size]]]
                        p_score, n_score_ = self.flen(task_, iseval, curr_rel, support_meta, istest, edge_labels)
                        n_score = torch.cat((n_score, n_score_), dim=1)
                else:
                    task_ = [task[0], task[1], task[2], [task[3][0][_:_+self.batch_size]]]
                    p_score, n_score_ = self.flen(task_, iseval, curr_rel, support_meta, istest, edge_labels)
                    n_score = torch.cat((n_score, n_score_), dim=1)

                y = torch.Tensor([1]).to(self.device).to(self.device).expand([p_score.shape[0], p_score.shape[1]])
                loss = self.flen.loss_func(p_score, n_score, y)

        return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        print('begin training')
        for e in range(self.epoch):
            # sample one batch from data_loader
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, _, _ = self.do_one_step(train_task, iseval=False, curr_rel=curr_rel, istest=False)
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
                self.wandb.log({'epoch':e, 'loss':loss_num})
                
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))

                valid_data,_,_ = self.eval_by_relation(istest=False, epoch=e)
                self.wandb.log({'epoch':e, 'Hits@10':valid_data['Hits@10']})
                if self.shuffle_background:
                    for ent in self.e1_rele2:
                        if len(self.e1_rele2[ent]) > self.max_neighbor:
                            random.shuffle(self.e1_rele2[ent])

                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    self.save_checkpoint()
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        print('Finish')
        return best_epoch

    def eval(self, istest=False, epoch=None):
        #self.flen.eval()
        # clear sharing rel_q
        self.flen.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        bi_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        n_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        all_t = 0
        bi_t = 0
        n_t = 0
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            len_x = len([_ for _ in eval_task[2][0][0] if _ != '[PAD]'])            
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            all_t += 1
            if len_x <= 3:
                bi_t += 1
            else:
                n_t += 1

            _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel, istest=istest)
            x = torch.cat([n_score, p_score], 1).squeeze()
            rank, pre_ent_ind = self.rank_predict(all_data, x, ranks, bi_data, n_data, len_x)

        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
            if bi_t!=0:
                bi_data[k] = round(bi_data[k] / bi_t, 3)
            if n_t!=0:
                n_data[k] = round(n_data[k] / n_t, 3)

        # print overall evaluation result and return it
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@3'], all_data['Hits@1']))
        print("Bi_data {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               bi_t, bi_data['MRR'], bi_data['Hits@10'], bi_data['Hits@5'], bi_data['Hits@3'], bi_data['Hits@1']))
        print("N_data {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               n_t, n_data['MRR'], n_data['Hits@10'], n_data['Hits@5'], n_data['Hits@3'], n_data['Hits@1']))

        return all_data, bi_data, n_data

    def case_eval(self, istest=False, epoch=None):
        #self.flen.eval()
        # clear sharing rel_q
        self.flen.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        bi_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        n_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        bi_t = 0
        n_t = 0
        f = open('out.txt', 'w')
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_case_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=curr_rel, istest=istest)
            x = torch.cat([n_score, p_score], 1).squeeze()

            len_x = len([_ for _ in eval_task[2][0][0] if _ != '[PAD]'])

            rank, pre_ent_ind = self.rank_predict(data, x, ranks, bi_data, n_data, len_x)
            if rank != 1:
                f.writelines('Hit:'+str(rank)+'pre_ent:'+eval_task[3][0][pre_ent_ind][2]+'Query:'+",".join(eval_task[2][0][0]).strip()+'\n')
            else:
                f.writelines('Hit:'+str(rank)+'pre_ent:'+eval_task[2][0][0][2]+'Query:'+",".join(eval_task[2][0][0]).strip()+'\n')

            if len_x <= 3:
                bi_t += 1
            else:
                n_t += 1

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)
            if bi_t!=0:
                bi_data[k] = round(bi_data[k] / bi_t, 3)
            if n_t!=0:
                n_data[k] = round(n_data[k] / n_t, 3)

        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@3'], data['Hits@1']))
        print("Bi_data {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               bi_t, bi_data['MRR'], bi_data['Hits@10'], bi_data['Hits@5'], bi_data['Hits@3'], bi_data['Hits@1']))
        print("N_data {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               n_t, n_data['MRR'], n_data['Hits@10'], n_data['Hits@5'], n_data['Hits@3'], n_data['Hits@1']))

        return data, bi_data, n_data

    def eval_by_relation(self, istest=False, epoch=None):
        #self.flen.eval()
        self.flen.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        bi_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        n_data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
        all_t = 0
        bi_t = 0
        n_t = 0
        all_ranks = []

        for rel in data_loader.all_rels:
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0

            ranks = []
            while True:
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                len_x = len([_ for _ in eval_task[2][0][0] if _ != '[PAD]'])            
                if eval_task == 'EOT':
                    break
                t += 1
                if len_x <= 3:
                    bi_t += 1
                else:
                    n_t += 1

                _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel, istest=istest)
                x = torch.cat([n_score, p_score], 1).squeeze()
                rank, pre_ent_ind = self.rank_predict(data, x, ranks, bi_data, n_data, len_x)

            for k in data.keys():
                temp[k] = data[k] / t
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                   rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))
            print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
                   t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@3'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
            if bi_t!=0:
                bi_data[k] = round(bi_data[k] / bi_t, 3)
            if n_t!=0:
                n_data[k] = round(n_data[k] / n_t, 3)
        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
            all_t, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@3'], all_data['Hits@1']))
        print("Bi_data {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               bi_t, bi_data['MRR'], bi_data['Hits@10'], bi_data['Hits@5'], bi_data['Hits@3'], bi_data['Hits@1']))
        print("N_data {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@3: {:.3f}\tHits@1: {:.3f}\r".format(
               n_t, n_data['MRR'], n_data['Hits@10'], n_data['Hits@5'], n_data['Hits@3'], n_data['Hits@1']))

        return all_data, bi_data, n_data