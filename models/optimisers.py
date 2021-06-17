import torch
from torchlars import LARS
import numpy as np


def get_optimiser(args, model):

    if args.optimiser == 'SGD':
        optimiser = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimiser == 'LARS':
        # optimised using LARS with square root rate scaling
        # the simCLR original paper best settings
        # here we are ignoring the learning rate set in the argparser
        learning_rate = 0.075 * np.sqrt(args.batch_size)
        base_optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        # keep the default LARs settings
        optimiser = LARS(optimizer=base_optimiser)
    else:
        raise NotImplementedError
    return optimiser
