_base_ = './default.py'

expname = f'384'
basedir = './logs/zju/6_views_FINAL'

data = dict(
    datadir='./data/zju/384/cache_train.pickle',
    dataset_type='zju',
    # Training data
    inverse_y=True,
    canonical_t=0.,
    video_len=756,
    flip_x=False,
    flip_y=False,
)
