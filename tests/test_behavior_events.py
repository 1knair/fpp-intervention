import numpy as np
import pandas as pd
from src.representation.event_features import detect,junctions,speed_transitions,summarize
from src.representation.segmentation import collision_avoidance

def telemetry():
    t=np.arange(0,20.001,.05);n=len(t)
    frame=pd.DataFrame({c:np.zeros(n)for c in ['Heading','Intersections','Collision','Signs','Events','Acceleration','Brake Pedal Force','Accelerator Pedal Position','Lane Offset','Heading Error','Steering Wheel Position','Vertical Velocity','Pitch','Roll','Headway Time','Headway Distance']})
    frame['X Position']=10*t;frame['Y Position']=0.;frame['Sim Time']=t;frame['Velocity']=10.;frame['Lane Number']=1.;frame['Speed Limit']=40.
    return frame

def test_physical_events_and_gap_rejection():
    t=telemetry();t.loc[t['Sim Time'].between(4,6),'Velocity']=0
    t.loc[t['Sim Time'].between(8,9),'Acceleration']=2
    t.loc[t['Sim Time'].between(10,10.2),'Collision']=1
    t.loc[t['Sim Time']>=12,'Lane Number']=2
    found=detect(t)
    assert set(['stop','acceleration','contact','lane_change']).issubset(found.kind)
    bad=t.copy();bad.loc[bad['Sim Time']>=5,'Sim Time']+=2
    assert detect(bad).query("kind=='stop'").empty
    assert summarize(bad)[1]=='clock_gap'

def test_heading_wrap_is_not_a_turn_and_speed_response():
    t=telemetry();t.loc[t['Sim Time']<10,'Heading']=359;t.loc[t['Sim Time']>=10,'Heading']=1
    t.loc[t['Sim Time'].between(9,11),'Intersections']=1
    assert junctions(t).maneuver.tolist()==['straight']
    t.loc[t['Sim Time']>=5,'Speed Limit']=20;t.loc[t['Sim Time']>=7,'Velocity']=8
    r=speed_transitions(t).iloc[0]
    assert r.direction=='down'and np.isclose(r.response_delay_s,2)

def test_event1_closure_is_not_collision():
    t=telemetry();t.loc[t['Sim Time'].between(7,9),'Events']=1
    t.loc[t['Sim Time'].between(5,5.2),'Signs']=-6
    t.loc[t['Sim Time'].between(10,10.2),'Signs']=-7
    assert collision_avoidance(t)==(None,'event1_is_lane_closure')
    t['Signs']=0
    assert collision_avoidance(t)[1]=='candidate'
