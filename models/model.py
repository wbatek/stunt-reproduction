from models.protonet_model.mlp import MLPProto

dataset_to_in = {
    'income': 105,
    'diabetes': 8,
    'optdigits': 64,
    'cmc': 24,
    'dna': 180,
    'karhunen': 64,
    'semeion': 256,
    'pixel': 240,
}


def get_model(P, modelstr):

    if modelstr == 'mlp':
        if 'protonet' in P.mode:
            # if P.dataset == 'income':
            #     model = MLPProto(105, 1024, 1024)
            # elif P.dataset == 'diabetes':
            #     model = MLPProto(8, 1024, 1024)
            # elif P.dataset == 'optdigits':
            #     model = MLPProto(64, 1024, 1024)
            # elif P.dataset == 'cmc':
            #     model = MLPProto(9, 1024, 1024)
            # elif P.dataset == 'dna':
            #     model = MLPProto(180, 1024, 1024)
            # elif P.dataset == 'karhunen':
            #     model = MLPProto(64, 1024, 1024)
            model = MLPProto(P.kernel_size, 1024, 1024)
    else:
        raise NotImplementedError()

    return model
