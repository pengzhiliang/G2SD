import argparse
import torch.nn as nn
from mmcv import Config
from mmcv.cnn import get_model_complexity_info #, get_model_parameters_number

def get_model_parameters_number(model: nn.Module) -> float:
    """Calculate parameter number of a model.
    Args:
        model (nn.module): The model for parameter number calculation.
    Returns:
        float: Parameter number of the model.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

from mmseg.models import build_segmentor
from backbone import mae, beit

def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    # flops, params = get_model_complexity_info(model, input_shape)
    params = get_model_parameters_number(model)
    split_line = '=' * 30
    # print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
    #     split_line, input_shape, flops, params))
    print('{0}\nInput shape: {1}\nParams: {2}\n{0}'.format(
        split_line, input_shape,  params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()