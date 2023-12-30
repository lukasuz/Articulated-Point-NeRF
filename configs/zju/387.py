_base_ = './default.py'

expname = f'387'
basedir = './logs/zju/6_views'

data = dict(
    datadir='./data/zju/387/cache_train.pickle',
    dataset_type='zju',
    # Training data
    inverse_y=True,
    canonical_t=0.,
    video_len=523,
    flip_x=False,
    flip_y=False,
)
