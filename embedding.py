import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter['device']
        self.ent2id = dataset['ent2id']
        self.rel2id = dataset['rel2id']
        self.es = parameter['embed_dim']
        self.max_seq_length = parameter['max_seq_length']
        self.fine_tune = parameter['fine_tune']

        num_ent = len(self.ent2id)
        num_rel = len(self.rel2id)
        self.rel_embedding = nn.Embedding(num_rel, self.es, padding_idx=0)
        self.ent_embedding = nn.Embedding(num_ent, self.es, padding_idx=0)

        if parameter['use_pretrain'] == 1:
            self.ent2emb = dataset['ent2emb']
            self.ent_embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
            self.rel2emb = dataset['rel2emb']
            self.rel_embedding.weight.data.copy_(torch.from_numpy(self.rel2emb))
        else:
            nn.init.xavier_uniform_(self.rel_embedding.weight)
            nn.init.xavier_uniform_(self.ent_embedding.weight)
        if not self.fine_tune and parameter['use_pretrain']:
            self.ent_embedding.weight.requires_grad = False
            self.rel_embedding.weight.requires_grad = False

    def forward(self, tuples):
        ht_idx = [[[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in tuples]
        ht_idx = torch.LongTensor(ht_idx).to(self.device)
        h_t_emb =  self.ent_embedding(ht_idx)
        
        qual_idx = [[[self.rel2id[t[_]] if _%2==1 else self.ent2id[t[_]] for _ in range(3, self.max_seq_length)] for t in batch] for batch in tuples]
        qual_idx = torch.LongTensor(qual_idx).to(self.device)
        qual_rel, qual_ent = qual_idx[:,:,::2], qual_idx[:,:,1::2]
        qual_rel_emb, qual_ent_emb = self.rel_embedding(qual_rel), self.ent_embedding(qual_ent)
        return h_t_emb, qual_rel_emb, qual_ent_emb