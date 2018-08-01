import argparse
parser = argparse.ArgumentParser(description="PyTorch Dureader")
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.1, help='rate of lr decay')

if __name__=="__main__":
    args = parser.parse_args()
    config = args.config
    print(type(config))
    print(config)
    print(type(args))
    print(args)
    print(args.config)
    args.mm ="ghjk"
    args.io=90
    print(args)