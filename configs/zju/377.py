_base_ = './default.py'

expname = f'377'
basedir = './logs/zju/'

data = dict(
    datadir='./data/zju/377/cache_train.pickle',
    dataset_type='zju',
    white_bkgd=False,
    # Training data
    inverse_y=True,
    canonical_t=0.,
    video_len=493,
    flip_x=False,
    flip_y=False,
)
