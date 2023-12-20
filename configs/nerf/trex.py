_base_ = './default.py'

expname = f'trex'
basedir = './logs/dnerf/'

data = dict(
    datadir='./data/dnerf/trex',
    dataset_type='dnerf',
    white_bkgd=True,
    canonical_t=0.25,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
    half_res=True,
)
