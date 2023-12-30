_base_ = './default.py'

expname = 'baxter'
basedir = './logs/wim/'

data = dict(
    datadir='./data/WIM/baxter',
    dataset_type='wim',
    canonical_t=0.035,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
)
