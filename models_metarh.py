from embedding import *
from hyper_embedding import *
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale, edge_key, edge_value, edge_mask):
        # edge_key L,L,H
        # q BN,L,H
        attn_score = torch.bmm(q, k.transpose(1, 2)) # BN,L,L
        edge_bias = torch.bmm(edge_key, q.permute(1,2,0)).transpose(0,2)
        attn_score += edge_bias
        if scale:
            attn_score = attn_score * scale
        # q M, B*N, H
        # edge_bias M, B*N, M
        # attn_score B,M,M
        # output B,M,H

        new_attn_mask = torch.zeros_like(edge_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(edge_mask, float("-inf"))
        edge_mask = new_attn_mask
        
        attn_score += edge_mask
        attn_score = self.softmax(attn_score)
        attn_score = self.dropout(attn_score)
        output = torch.bmm(attn_score, v)

        # attn B, M, M->M,B,M
        # edge_value M,M,H
        # edge_bias M,B,H->B,M,H
        edge_bias = torch.bmm(attn_score.transpose(0, 1), edge_value).transpose(0,1)
        output += edge_bias
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.linear_final = nn.Linear(embed_dim, embed_dim)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, edge_value, edge_key, edge_mask):
        max_seq_length, batch_size, embed_dim = key.size()
        residual = query
        head_dim = embed_dim // self.num_heads
        key = self.linear_k(key)
        query = self.linear_q(query)
        value = self.linear_v(value)

        # M, B*N, H
        key = key.contiguous().view(max_seq_length, batch_size*self.num_heads, head_dim).transpose(0,1)
        query = query.contiguous().view(max_seq_length, batch_size*self.num_heads, head_dim).transpose(0,1)
        value = value.contiguous().view(max_seq_length, batch_size*self.num_heads, head_dim).transpose(0,1)

        scale = float(head_dim) ** -0.5
        context = self.dot_product_attention(query, key, value, scale, edge_key, edge_value, edge_mask)
        context = context.transpose(0,1).contiguous().view(max_seq_length, batch_size, embed_dim)

        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(output+residual)

        return output


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        fnn_dim = 2048
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(embed_dim, fnn_dim)
        self.linear2 = nn.Linear(fnn_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, context):
        context = self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(context))))
        context = context + self.dropout(context)
        context = self.layer_norm(context)
        return context

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(embed_dim)
    
    def forward(self, inputs, edge_value, edge_key, edge_mask):
        output = self.attention(inputs, inputs, inputs, edge_value, edge_key, edge_mask)
        output = self.feed_forward(output)
        return output

class GraphTransformerEncoder(nn.Module):
    def __init__(self, num_heads=4, num_transformer_layers=12, embed_dim=100, dropout=0.2):
        super().__init__()
        num_layers = num_transformer_layers
        self.num_heads = num_heads
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embed_dim, self.num_heads, dropout) for _ in range(num_layers)]
        )
    def forward(self, facts, edge_value, edge_key, edge_mask):
        facts = facts.transpose(0, 1)
        # L, B, H
        for encoder in self.encoder_layers:
            facts = encoder(facts, edge_value, edge_key, edge_mask)
        return facts

def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    size = a.shape
    a = a.view(size[0]*size[1]*size[2], size[3])
    b = b.view(size[0]*size[1]*size[2], size[3])
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1,
                       signal_sizes=(a.shape[-1],)).view(size[0], size[1], size[2], size[3])

def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                        h_re * r_im + h_im * r_re], dim=-1)

class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, device=None):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.lstm = nn.LSTM(self.embed_size*2, self.n_hidden, self.layers, bidirectional=True)
        self.out = nn.Linear(self.n_hidden*2*self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden*2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1,2), soft_attn_weight)
        context = context.view(-1, self.n_hidden*2*self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden).to(self.device))
        cell_state = Variable(torch.zeros(self.layers*2, size[0], self.n_hidden).to(self.device))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_hidden_state)

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)



class Flen(nn.Module):
    def __init__(self, dataset, parameter, embed = None):
        super(Flen, self).__init__()
        self.device = parameter['device']
        self.max_seq_length = parameter['max_seq_length']
        self.beta = parameter['beta']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.use_neighbors = parameter['use_neighbors']
        self.use_pretrain = parameter['use_pretrain']
        self.fine_tune = parameter['fine_tune']
        self.abla = parameter['ablation']
        # 只包含实体的embed
        self.embedding = Embedding(dataset, parameter)

        self.few = parameter['few']
        self.dropout_i = nn.Dropout(parameter['dropout_i'])


        if self.use_neighbors:
            self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)
            self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
            self.attn_w = nn.Linear(self.embed_dim, 1)

            self.gate_w = nn.Linear(self.embed_dim, 1)
            self.gate_b = nn.Parameter(torch.FloatTensor(1))

            init.xavier_normal_(self.gcn_w.weight)
            init.xavier_normal_(self.attn_w.weight)
            init.constant_(self.gcn_b, 0)
            init.constant_(self.gate_b, 0)

        self.rel_w = parameter['weight']

        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()

        self.relation_learner = parameter['relation_learner']
        self.embedding_learner = parameter['embedding_learner']

        self.h_norm = None
        self.h_embedding = H_Embedding(dataset, parameter)

        # num_layers, num_heads, ffn_dim, dropout = 2, 4, 2000, 0.1
        self.num_layers = parameter['num_layers']
        self.num_heads = parameter['num_heads']
        self.dropout_g = parameter['dropout_g']
        if self.relation_learner == 'transformer':
            encoder_layers = TransformerEncoderLayer(self.embed_dim, self.num_heads)
            self.TruTransformerEncoder = TransformerEncoder(encoder_layers, self.num_layers)
            self.position_embeddings = nn.Embedding(5, self.embed_dim)
            self.position_embeddings.weight.requires_grad = False
        elif self.relation_learner == 'gran':
            self.n_edge = 5
            self.edge_key_embedding = nn.Embedding(self.n_edge, self.embed_dim//self.num_heads, padding_idx=0)
            self.edge_value_embedding = nn.Embedding(self.n_edge, self.embed_dim//self.num_heads, padding_idx=0)

            self.GraphTransformerEncoder = GraphTransformerEncoder(
                                                num_heads=self.num_heads,
                                                num_transformer_layers=self.num_layers, 
                                                embed_dim=self.embed_dim,
                                                dropout=self.dropout_g)


        self.object_mask_emb = torch.nn.Parameter(torch.randn(1, self.embed_dim, dtype=torch.float32),True)

        # ccorr, sub, mult, rotate
        self.qual_opn = parameter['qual_opn']
        # sum, mean
        self.qual_n = parameter['qual_n']
        # sum, concat, mul
        self.qual_aggregate = parameter['qual_aggregate']
        # sum, min, max, mean, concat
        self.few_rel_aggregate = parameter['few_rel_aggregate']
        if self.qual_aggregate == 'concat':
            self.w_q = nn.Parameter(torch.Tensor(self.embed_dim*2, self.embed_dim))
        elif self.qual_aggregate in ['sum', 'mul']:
            self.w_q = nn.Parameter(torch.Tensor(self.embed_dim, self.embed_dim))
        if self.few_rel_aggregate == 'concat':
            self.w_few_rel = nn.Parameter(torch.Tensor(self.embed_dim*self.few, self.embed_dim))
            init.xavier_normal_(self.w_few_rel.data)
        init.xavier_normal_(self.w_q.data)
        if self.few_rel_aggregate == 'attn':
            self.few_q = nn.Parameter(torch.Tensor(self.embed_dim, 1))
            init.xavier_normal_(self.few_q.data)


    def Embedding_learner(self, h, t, q, r, pos_num, norm=None):
        if self.qual_aggregate == 'sum':
            q = torch.einsum('ijkd, dd -> ijkd', q, self.w_q)
            r = self.rel_w*r + (1-self.rel_w)*q
        elif self.qual_aggregate == 'concat':
            r = torch.cat((r, q), dim=-1)
            r = torch.einsum('ijkd, db -> ijkb', r, self.w_q)
        elif self.qual_aggregate == 'mul':
            size = q.shape
            q = q.view(size[0]*size[1]*size[2], size[3])
            mask = q.sum(dim=-1) == 0
            # q[mask] = 1
            for i in range(len(mask)):
                if mask[i]:
                    q[i,:] = 1
            q = q.view(size[0], size[1], size[2], size[3])
            q = torch.einsum('ijkd, dd -> ijkd', r, self.w_q)
            r = r*q
        if self.embedding_learner == 'mtransh':
            norm = norm[:,:1,:,:]						# revise
            h = h - torch.sum(h * norm, -1, True) * norm
            t = t - torch.sum(t * norm, -1, True) * norm
            score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        elif self.embedding_learner == 'transe':
            score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score

    def neighbor_encoder(self, connections, entself_embeds):
        '''
        connections: (batch, 200, 2)
        '''
        # 512, 50, 16
        relations = connections[:,:,0].squeeze(-1)
        entities = connections[:,:,1].squeeze(-1)
        rel_embeds = self.dropout_i(self.embedding.rel_embedding(relations)) # (batch, 200, embed_dim)
        ent_embeds = self.dropout_i(self.embedding.ent_embedding(entities)) # (batch, 200, embed_dim)

        qualifier = connections[:,:,2:]
        qualifier_rel = qualifier[:,:,::2]
        qualifier_ent = qualifier[:,:,1::2]
        qual_rel_embeds = self.dropout_i(self.embedding.rel_embedding(qualifier_rel))
        qual_ent_embeds = self.dropout_i(self.embedding.ent_embedding(qualifier_ent))

        if self.qual_opn == 'sub':
            qual_embeds = qual_ent_embeds-qual_rel_embeds
        elif self.qual_opn == 'ccorr':
            qual_embeds = ccorr(qual_ent_embeds,qual_rel_embeds)
        elif self.qual_opn == 'mult':
            qual_embeds = qual_ent_embeds*qual_rel_embeds
        elif self.qual_opn == 'rotate':
            qual_embeds = rotate(qual_ent_embeds,qual_rel_embeds)
        
        if self.qual_n == 'sum':
            qual_embeds = torch.sum(qual_embeds, dim=2)
        elif self.qual_n == 'mean':
            qual_embeds = torch.mean(qual_embeds, dim=2)

        if self.qual_aggregate == 'sum':
            qual_embeds = torch.einsum('ijd, dd -> ijd', qual_embeds, self.w_q)
            # qual_embeds = torch.matmul(qual_embeds, self.w_q)
            rel_embeds = self.rel_w*rel_embeds + (1-self.rel_w)*qual_embeds
        elif self.qual_aggregate == 'concat':
            rel_embeds = torch.cat((rel_embeds, qual_embeds), dim=-1)
            rel_embeds = torch.einsum('ijd, db -> ijb', rel_embeds, self.w_q)
            # rel_embeds = torch.matmul(rel_embeds, self.w_q)
        elif self.qual_aggregate == 'mul':
            size = qual_embeds.shape
            qual_embeds = qual_embeds.view(size[0]*size[1], size[2])
            mask = qual_embeds.sum(dim=-1) == 0
            for i in range(len(mask)):
                if mask[i]:
                    qual_embeds[i,:] = 1
            # qual_embeds[mask] = 1
            qual_embeds = qual_embeds.view(size[0], size[1], size[2])
            qual_embeds = torch.einsum('ijd, dd -> ijd', qual_embeds, self.w_q)
            # qual_embeds = torch.matmul(qual_embeds, self.w_q)
            rel_embeds = rel_embeds*qual_embeds
        
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1) # (batch, 200, 2*embed_dim)
        out = self.gcn_w(concat_embeds) + self.gcn_b
        out = F.leaky_relu(out)
        attn_out = self.attn_w(out)
        attn_weight = F.softmax(attn_out, dim=1)
        out_attn = torch.bmm(out.transpose(1,2), attn_weight)
        
        out_attn = out_attn.squeeze(2)
        # 门控机制使用到了sigmod
        gate_tmp = self.gate_w(out_attn) + self.gate_b
        gate = torch.sigmoid(gate_tmp)
        out_neigh = torch.mul(out_attn, gate)
        out_neighbor = out_neigh + torch.mul(entself_embeds,1.0-gate)
        return out_neighbor


    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        pos_neg_qual = torch.cat([positive[:, :, 2, :],
                                negative[:, :, 2, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2, pos_neg_qual

    def qualifier_embedding(self, qual_rel_embeds, qual_ent_embeds, tuples):
        idx = [[[self.embedding.rel2id[t[_]] if _%2==1 else self.embedding.ent2id[t[_]] for _ in range(3, self.max_seq_length)] for t in batch] for batch in tuples]
        qualifier_emb = torch.cat((qual_rel_embeds, qual_ent_embeds), dim=3)
        qualifier_emb = qualifier_emb.view(qual_ent_embeds.shape[0], qual_ent_embeds.shape[1],qual_ent_embeds.shape[2]*2,qual_ent_embeds.shape[3])

        if self.qual_opn == 'sub':
            qual_embeds = qual_ent_embeds-qual_rel_embeds
        elif self.qual_opn == 'ccorr':
            qual_embeds = ccorr(qual_ent_embeds,qual_rel_embeds)
        elif self.qual_opn == 'mult':
            qual_embeds = qual_ent_embeds*qual_rel_embeds
        elif self.qual_opn == 'rotate':
            qual_embeds = rotate(qual_ent_embeds,qual_rel_embeds)

        if self.qual_n == 'sum':
            qual_embeds = torch.sum(qual_embeds, dim=2)
        elif self.qual_n == 'mean':
            qual_embeds = torch.mean(qual_embeds, dim=2)

        mask = torch.zeros(qualifier_emb.shape[0], qualifier_emb.shape[1], self.max_seq_length).bool().to(self.device)
        
        mask[:, :, 3:] = torch.Tensor(idx).to(self.device) == 0
        return qual_embeds, qualifier_emb, mask

    def forward(self, task, iseval=False, curr_rel='', support_meta=None, istest=False, edge_labels=False):
        # transfer task string into embedding
        # 这里是ent2id
        # support/neighbor都是使用rel2id、ent2id
        query, query_qual_rel, query_qual_ent = self.embedding(task[2])
        negative, negative_qual_rel, negative_qual_ent = self.embedding(task[3])
        query_q, _, _ = self.qualifier_embedding(query_qual_rel, query_qual_ent, task[2])
        negative_q, _, _ = self.qualifier_embedding(negative_qual_rel, negative_qual_ent, task[3])
        query = torch.cat((query, query_q.unsqueeze(2)), dim=2)
        negative = torch.cat((negative, negative_q.unsqueeze(2)), dim=2)
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative


        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            norm_vector = self.h_embedding(task[0])

            support, support_qual_rel, support_qual_ent = self.embedding(task[0])
            support_negative, support_negative_qual_rel, support_negative_qual_ent = self.embedding(task[1])
            support_q, support_qual, support_mask = self.qualifier_embedding(support_qual_rel, support_qual_ent, task[0])
            support_negative_q, support_negative_qual, _ = self.qualifier_embedding(support_negative_qual_rel, support_negative_qual_ent, task[1])

            support = torch.cat((support, support_q.unsqueeze(2)), dim=2)
            support_negative = torch.cat((support_negative,support_negative_q.unsqueeze(2)), dim=2)
            object_mask = self.object_mask_emb.repeat(support_q.shape[0], 1).to(self.device)

            num_sn = support_negative.shape[1]  # num of support negative

            # 得到背景增强后的支持集第一个样例中的关系表示
            support_left_connections, support_right_connections = support_meta[0]
            if self.use_neighbors:
                support_left = self.neighbor_encoder(support_left_connections, support[:,0, 0, :]) #512, 5, 3, 200
                support_right = self.neighbor_encoder(support_right_connections, support[:,0, 1, :])
            else:
                support_left = support[:,0, 0, :]
                support_right = support[:,0, 1, :]
            support_few = torch.cat((object_mask.unsqueeze(1), support_left.unsqueeze(1), support_right.unsqueeze(1), support_qual[:,0,:,:]), dim=1).unsqueeze(1)

            for i in range(self.few-1):
                support_left_connections, support_right_connections = support_meta[i+1]
                if self.use_neighbors:
                    support_left = self.neighbor_encoder(support_left_connections, support[:,i+1, 0, :])
                    support_right = self.neighbor_encoder(support_right_connections, support[:,i+1, 1, :])
                else:
                    support_left = support[:,i+1, 0, :]
                    support_right = support[:,i+1, 1, :]
                support_pair = torch.cat((object_mask.unsqueeze(1), support_left.unsqueeze(1), support_right.unsqueeze(1), support_qual[:,i+1,:,:]), dim=1).unsqueeze(1)  # tanh
                support_few = torch.cat((support_few, support_pair), dim=1)
            # 得到背景增强后的全部支持集的小样本关系的表示
            support_few = support_few.view(support.shape[0]*self.few, self.max_seq_length, self.embed_dim)
            support_mask = support_mask.view(support.shape[0]*self.few, self.max_seq_length)
            
            # split on e1/e2 and concat on pos/neg
            # 学习支持集中的关系表示
            # positions = torch.arange(support_few.shape[1], dtype=torch.long, device=self.device).repeat(support_few.shape[0], 1)
            if self.relation_learner == 'transformer':
                positions = torch.LongTensor([0,1,2]+[3,4]*((self.max_seq_length-3)//2)).to(self.device).repeat(support_few.shape[0], 1)
                positions_emb = self.position_embeddings(positions)
                support_few += positions_emb
                rel = self.TruTransformerEncoder(support_few.transpose(1,0), src_key_padding_mask=support_mask)[0]
            elif self.relation_learner == 'gran':
                edge_mask = torch.bmm(support_mask.unsqueeze(2).float(), support_mask.unsqueeze(1).float())
                # edge_mask = edge_mask*-1000000.0
                edge_mask = edge_mask.repeat(1, 1, self.num_heads).view(self.num_heads*support_mask.shape[0], support_mask.shape[1], support_mask.shape[1]).bool()
                # edge_mask = edge_mask.repeat(self.num_heads,1,1)
                edge_value = self.edge_value_embedding(edge_labels)
                edge_key = self.edge_key_embedding(edge_labels)
                rel = self.GraphTransformerEncoder(
                    support_few,
                    edge_value,
                    edge_key,
                    edge_mask,
                    )[0]
            rel = rel.view(support.shape[0], self.few, 1, self.embed_dim)

            if self.few_rel_aggregate == 'mean':
                rel = torch.mean(rel, dim=1).unsqueeze(1)
            elif self.few_rel_aggregate == 'max':
                rel = torch.max(rel, dim=1).unsqueeze(1)
            elif self.few_rel_aggregate == 'min':
                rel = torch.min(rel, dim=1).unsqueeze(1)
            elif self.few_rel_aggregate == 'sum':
                rel = torch.sum(rel, dim=1).unsqueeze(1)
            elif self.few_rel_aggregate == 'concat':
                rel = rel.view(support.shape[0], self.few*self.embed_dim)
                rel = torch.einsum('ik, kd -> id', rel, self.w_few_rel)
                rel = rel.view(support.shape[0], 1, 1, self.embed_dim)
            elif self.few_rel_aggregate == 'attn':
                attn_weight = torch.einsum('ikwj, jw -> ik', rel, self.few_q)
                soft_attn_weight = F.softmax(attn_weight, 1)
                # 1024 5 100   ->  1024 5 1
                rel = torch.einsum('ikwj, ik -> ij', rel, soft_attn_weight)
                rel = rel.view(support.shape[0], 1, 1, self.embed_dim)

            if not self.abla:
                rel.retain_grad()

                # relation for support
                sup_neg_e1, sup_neg_e2, sup_neg_qual = self.split_concat(support, support_negative)
                rel_s = rel.expand(-1, self.few+num_sn, -1, -1)
                p_score, n_score = self.Embedding_learner(sup_neg_e1, sup_neg_e2, sup_neg_qual, rel_s, self.few, norm_vector)	# revise norm_vector

                y = torch.Tensor([[1]]).to(self.device).expand([p_score.shape[0], 1])
                self.zero_grad()
                loss = self.loss_func(p_score, n_score, y)
                loss.backward(retain_graph=True)
                grad_meta = rel.grad
                rel_q = rel - self.beta*grad_meta
                norm_q = norm_vector - self.beta*grad_meta				# hyper-plane update
            else:
                rel_q = rel
                norm_q = norm_vector

            self.rel_q_sharing[curr_rel] = rel_q

            self.h_norm = norm_vector.mean(0)
            self.h_norm = self.h_norm.unsqueeze(0)

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)

        que_neg_e1, que_neg_e2, que_neg_qual = self.split_concat(query, negative)  # [bs, nq+nn, 1, es]
        norm_q = self.h_norm
        p_score, n_score = self.Embedding_learner(que_neg_e1, que_neg_e2, que_neg_qual, rel_q, num_q, norm_q)

        return p_score, n_score
