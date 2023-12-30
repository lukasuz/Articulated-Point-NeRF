_base_ = './default.py'

expname = f'336'
basedir = './logs/zju/'

data = dict(
    datadir='/home/lukas/projects/watch-it-move/data/zju_mocap/cache512_6_views/366/cache_train.pickle',
    dataset_type='zju',
    # Training data
    canonical_t=0.,
    video_len=623,
    inverse_y=True,
    flip_x=False,
    flip_y=False,
)
