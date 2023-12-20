_base_ = './default.py'

expname = 'pandas'
basedir = './logs/wim/'

data = dict(
    datadir='./data/WIM/pandas',
    dataset_type='wim',
    white_bkgd=True,
    canonical_t=0.96,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
