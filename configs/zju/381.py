_base_ = './default.py'

expname = f'381'
basedir = './logs/zju/'

data = dict(
    datadir='./data/zju/381/cache_train.pickle',
    dataset_type='zju',
    # Training data
    inverse_y=True,
    canonical_t=0.,
    video_len=500,
    flip_x=False,
    flip_y=False,
)
