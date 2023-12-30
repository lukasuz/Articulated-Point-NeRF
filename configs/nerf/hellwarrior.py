_base_ = './default.py'

expname = f'hellwarrior'
basedir = './logs/dnerf/'

data = dict(
    datadir='./data/dnerf/hellwarrior',
    dataset_type='dnerf',
    canonical_t=0.,
    inverse_y=False,
    flip_x=False,
    flip_y=False,
    half_res=True,
)
