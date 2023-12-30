_base_ = './default.py'

expname = f'hook'
basedir = './logs/dnerf/'

data = dict(
    datadir='./data/dnerf/hook',
    dataset_type='dnerf',
    canonical_t=0.5,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
    half_res=True,
)
