_base_ = './default.py'

expname = f'standup'
basedir = './logs/dnerf/'

data = dict(
    datadir='./data/dnerf/standup',
    dataset_type='dnerf',
    white_bkgd=True,
    canonical_t=1.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
    half_res=True,
)
