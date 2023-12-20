_base_ = './default.py'

expname = 'iiwa'
basedir = './logs/wim/'

data = dict(
    datadir='./data/WIM/iiwa',
    dataset_type='wim',
    white_bkgd=True,
    canonical_t=0.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
