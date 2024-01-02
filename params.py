import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="F-JF17K", type=str)  # ["NELL-One", "Wiki-One"]
    args.add_argument("-path", "--data_path", default="Dataset/", type=str)  # ["./NELL", "./Wiki"]
    args.add_argument("-u_pretrain", "--use_pretrain", default=1, type=int)
    args.add_argument("-u_neighbors", "--use_neighbors", default=1, type=int)
    args.add_argument("-u_in_train", "--use_in_train", default=1, type=int)
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-few", "--few", default=5, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="Hits@10", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])
    
    args.add_argument("-e_model", "--embed_model", default="HINGE", type=str)
    args.add_argument("-m_seq_length", "--max_seq_length", default=11, type=int)

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)	# 0.001
    args.add_argument("-weight_decay", "--weight_decay", default=0.0001, type=float)	# 0.001
    args.add_argument("-es_p", "--early_stopping_patience", default=10, type=int)

    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)	# 5
    args.add_argument("-m", "--margin", default=1.0, type=float)	# default: 1
    args.add_argument("-i", "--dropout_i", default=0.5, type=float)
    args.add_argument("-p", "--dropout_g", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)
    args.add_argument("-fine_tune", "--fine_tune", default=1, type=int)

    args.add_argument("-n_layers", "--num_layers", default=2, type=int)
    args.add_argument("-n_heads", "--num_heads", default=4, type=int)

    # transformer, gran
    args.add_argument("-relation_learner", "--relation_learner", default='transformer', type=str)
    # transe, mtransh
    args.add_argument("-embedding_learner", "--embedding_learner", default='mtransh', type=str)
    args.add_argument("-shuffle_background", "--shuffle_background", default=1, type=int)

    args.add_argument("-gpu", "--device", default='0', type=str)
    args.add_argument("-weight", "--weight", default=1, type=float)
    # ccorr, sub, mult, rotate
    args.add_argument("-qual_opn", "--qual_opn", default='rotate', type=str)
    # sum, mean
    args.add_argument("-qual_n", "--qual_n", default='rotate', type=str)
    # sum, concat, mul
    args.add_argument("-qual_aggregate", "--qual_aggregate", default='sum', type=str)
    # sum, min, max, mean
    args.add_argument("-few_rel_aggregate", "--few_rel_aggregate", default='mean', type=str)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev', 'case'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-max_neighbor", "--max_neighbor", default=10, type=int)

    args = args.parse_args()

    args.data_path = args.data_path[:-1]+'/'
    params = {}
    for k, v in vars(args).items():
        params[k] = v



    return params


data_dir = {
    'train_tasks': '/train_tasks.json',
    'background': '/background.json',
    'test_tasks': "/test_tasks.json",
    'dev_tasks': "/dev_tasks.json",

    'rel2candidates_in_train': '/rel2candidates_in_train.json',
    'rel2candidates': '/rel2candidates.json',

    'Ground': '/Ground.json',
    'ents': '/ents.txt',
    'rels': '/rels.txt',
}
