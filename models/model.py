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
    'magic_telescope': 10,
    'marketing': 42,
    'electricity': 13,
    'nomao': 174,
    'gas-drift': 128,
    'morphological': 6,
    'blood-transfusion': 4,
    'balance-scale': 4,
    'pc1': 21,
    'qsar': 41,
    'pc4': 37,
    'krvskp': 73,
    'letter': 16,
    'factors': 216,
    'zernike': 47
}


def get_model(P, modelstr):

    if modelstr == 'mlp':
        if 'protonet' in P.mode:
            model = MLPProto(dataset_to_in[P.dataset], 1024, 1024)
    else:
        raise NotImplementedError()

    return model
