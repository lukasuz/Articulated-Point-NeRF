_base_ = './default.py'

expname = 'spot'
basedir = './logs/wim/'

data = dict(
    datadir='/home/lukas/data/WIM/spot/spot',
    dataset_type='wim',
    canonical_t=0.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
