import numpy as np
import pandas as pd
import pytest
from src.representation.laneclosure_features import calc_lane_closure_features, window_weights


def example():
    t=np.arange(0,20.01,.05)
    v=np.where(t<10,10.,5.)
    x=np.where(t<=10,10*t,100+5*(t-10))
    return pd.DataFrame({'Sim Time':t,'Velocity':v,'X Position':x,'Y Position':0.,
        'Lane Number':np.where(t<9,1.,2.),'Lane Offset':0.,'Collision':0.})


def test_signed_timing_and_speed_reduction():
    values=calc_lane_closure_features(example(),8,10,16,2,9)
    assert values['completion_lead_s']==pytest.approx(1)
    assert values['completion_lead_m']==pytest.approx(10)
    assert values['speed_reduction_mph']==pytest.approx(5/.44704)
    assert values['outside_corridor_fraction']==0
    assert values['return_in_window']==0


def test_persistent_return_and_sign_flip():
    telem=example()
    telem.loc[telem['Sim Time'].ge(14),'Lane Number']=1
    values=calc_lane_closure_features(telem,8,10,16,2,9)
    assert values['return_in_window']==1
    assert values['outside_corridor_fraction']==pytest.approx(2/6)
    telem=example()
    telem.loc[telem['Sim Time'].ge(14),'Lane Number']=-2
    assert calc_lane_closure_features(telem,8,10,16,2,9)['return_in_window']==0


def test_missing_completion_is_not_zero_lead():
    values=calc_lane_closure_features(example(),8,10,16,2,np.nan)
    assert values['no_completion']==1
    assert np.isnan(values['completion_lead_m'])
    assert np.isnan(values['return_in_window'])


def test_partial_sample_weights_and_clock_gap():
    weights=window_weights([0,.05,.10,.15],.025,.125)
    np.testing.assert_allclose(weights,[.025,.05,.025,0])
    with pytest.raises(ValueError,match='clock gap'):
        window_weights([0,.05,.3,.35],.025,.325)


def test_wrong_lane_can_have_zero_lane_offset():
    telem=example();telem['Lane Number']=1
    values=calc_lane_closure_features(telem,8,10,16,2,None)
    assert values['outside_corridor_fraction']==1
