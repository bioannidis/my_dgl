import torch as th
import torch.nn as nn

import dgl

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 embed_per_feat_size=16,
                 dgl_sparse=False,
                 per_feat_name_embed=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse
        self.embed_per_feat_size=embed_per_feat_size
        self.per_feat_name_embed=per_feat_name_embed

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.projection_matrix = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                    if dgl_sparse:
                        self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                            init_func=initializer)
                    else:
                        sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                        nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                        self.node_embeds[str(ntype)] = sparse_emb
            else:
                    if self.per_feat_name_embed:
                        # need a single embedding per node feat name
                        i=0
                        for feat in input_size[ntype]:
                            self.embeds[str(tuple((ntype,i)))] = nn.Parameter(th.Tensor(feat.shape[1], self.embed_per_feat_size))
                            nn.init.xavier_uniform_(self.embeds[str(tuple((ntype,i)))])
                            i+=1
                        self.projection_matrix[str(ntype)] = nn.Parameter(th.Tensor(self.embed_per_feat_size *i, self.embed_size))
                    else:
                        input_emb_size = 0
                        for feat in input_size[ntype]:
                            input_emb_size += feat.shape[1]
                        embed = nn.Parameter(th.Tensor(input_emb_size, self.embed_size))
                        nn.init.xavier_uniform_(embed)
                        self.embeds[str(ntype)] = embed


    @property
    def dgl_emb(self):
        # TODO how to handle this for the multiple feature sizes?
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.dev_id)
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)
        for ntype in range(self.num_of_ntype):
            loc = node_tids == ntype
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.dev_id)
            else:
                if self.per_feat_name_embed:
                    i=0
                    embeds_per_feat=[]
                    for feat in features[ntype]:
                        embeds_per_feat += [feat[type_ids[loc]].to(self.dev_id) @ self.embeds[str(tuple((ntype,i)))].to(
                            self.dev_id)]
                        i+=1
                    embeds_concated = th.cat(embeds_per_feat, dim=1).to(self.dev_id)
                    embeds[loc]=embeds_concated@ self.projection_matrix[str(ntype)].to(self.dev_id)
                else:
                    features_concated=th.cat(features[ntype], dim=1)
                    embeds[loc] = features_concated[type_ids[loc]].to(self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)

        return embeds


class EmbedCatBlock(nn.Module):
    """ Used to embed a single embedding feature. """
    def __init__(self, embed_dim, num_categories, **kwargs):
        super(EmbedCatBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.body = th.nn.Embedding(num_categories, embed_dim)  # for Xavier-style: scale = np.sqrt(3/float(embed_dim))
            nn.init.uniform_(self.body.weight, -1.0, 1.0)
    def hybrid_forward(self, x):
        return self.body(x)


class NumericBlock(nn.Module):
    """ Single Dense layer that jointly embeds all numeric and one-hot features """

    def __init__(self, embed_dim,input_size, **kwargs):
        super(NumericBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.body =nn.Linear(input_size, embed_dim)
    def hybrid_forward(self, x):
        return self.body(x)
class CustomRelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 embed_per_feat_size=16,
                 dgl_sparse=False,
                 per_feat_name_embed=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse
        self.embed_per_feat_size=embed_per_feat_size
        self.per_feat_name_embed=per_feat_name_embed

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.projection_matrix = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                    if dgl_sparse:
                        self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                            init_func=initializer)
                    else:
                        sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                        nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                        self.node_embeds[str(ntype)] = sparse_emb
            else:
                    if self.per_feat_name_embed:
                        # need a single embedding per node feat name
                        i=0
                        for feat in input_size[ntype]:
                            # TODO Concatenate all num features together before embedding layer.
                            feat_type,feat_vec=feat
                            if feat_type=='cat':
                                # The following counts the unique categories in the features to initialize the embedding.
                                self.embeds[str(tuple((ntype,i)))] =EmbedCatBlock(self.embed_size,len(th.unique(feat_vec)))
                            elif feat_type=="num":
                                self.embeds[str(tuple((ntype, i)))] = NumericBlock(self.embed_size,feat_vec.shape[0])
                            i+=1
                        self.projection_matrix[str(ntype)] = nn.Parameter(th.Tensor(self.embed_per_feat_size *i, self.embed_size))
                    else:
                        input_emb_size = 0
                        for feat in input_size[ntype]:
                            input_emb_size += feat.shape[1]
                        embed = nn.Parameter(th.Tensor(input_emb_size, self.embed_size))
                        nn.init.xavier_uniform_(embed)
                        self.embeds[str(ntype)] = embed


    @property
    def dgl_emb(self):
        # TODO how to handle this for the multiple feature sizes?
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.dev_id)
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)
        for ntype in range(self.num_of_ntype):
            loc = node_tids == ntype
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.dev_id)
            else:
                if self.per_feat_name_embed:
                    i=0
                    embeds_per_feat=[]
                    for feat in features[ntype]:
                        embeds_per_feat += [self.embeds[str(tuple((ntype,i)))](feat[type_ids[loc]].to(self.dev_id)).to(
                            self.dev_id)]
                        i+=1
                    embeds_concated = th.cat(embeds_per_feat, dim=1).to(self.dev_id)
                    embeds[loc]=embeds_concated@ self.projection_matrix[str(ntype)].to(self.dev_id)
                else:
                    features_concated=th.cat(features[ntype], dim=1)
                    embeds[loc] = features_concated[type_ids[loc]].to(self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)

        return embeds
def getEmbedSizes(size_factor,max_embedding_dim,embed_exponent, num_categs_per_feature):
    """ Returns list of embedding sizes for each categorical variable.
        Selects this adaptively based on training_datset.
        Note: Assumes there is at least one embed feature.
    """
    embed_dims = [int(size_factor*max(2, min(max_embedding_dim,
                                      1.6 * num_categs_per_feature[i]**embed_exponent)))
                   for i in range(len(num_categs_per_feature))]
    return embed_dims