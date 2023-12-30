_base_ = './default.py'

expname = f'lego'
basedir = './logs/dnerf/'

data = dict(
    datadir='./data/dnerf/lego',
    dataset_type='dnerf',
    canonical_t=1.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
    half_res=True,
)
