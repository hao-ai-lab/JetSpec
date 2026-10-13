"""Opt-in static routing from a measured fair matrix (J counts refinements)."""
import os
from typing import NamedTuple
import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest

class Route(NamedTuple):
    hypotheses: int
    branches: int
    passes: int
    per_sequence: bool
    fusion: str
    source: str

# Internal passes include the first draft: public J0 means passes=1.
# Fusion's FP32 identity is certified separately; the routes below are measured combinations.
SOURCE='metrics/the-l3-fair-matrix_w247_a100.csv'
ROUTES={
    1: Route(7,1,1,False,'all',SOURCE+' bs1 gsm8k top-7 F1 J1 P0'),
    4: Route(7,1,1,True,'all',SOURCE+' bs4 gsm8k/math500 top-7 F1 J1 P1'),
    16: Route(1,1,1,True,'all',SOURCE+' bs16 gsm8k/math500 one-k F1 J1 P1'),
}



def apply_auto_route(batch_size):
    """Install one recipe in each process before allocating its protocol wire.

    AUTO owns these recipe settings when enabled; when disabled it is inert.
    The main model/checkpoint and waiting protocol remain caller configuration.
    """
    if os.environ.get('JETSPEC_AUTO','0')!='1':
        return None
    route=select_route(batch_size,os.environ.get('JETSPEC_AUTO_PROFILE','default'))
    os.environ.update(JETSPEC_ROWS='1',JETSPEC_ROWS_FAST='1',JETSPEC_ROWS_GRAPHS='1',
        JETSPEC_F=str(route.branches),JETSPEC_PERSEQ=str(int(route.per_sequence)),
        JETSPEC_FUSE=route.fusion,JETSPEC_KSET={1:'one-k',7:'top-m',16:'all-k'}[route.hypotheses],
        JETSPEC_KSET_M=str(route.hypotheses),JETSPEC_KSET_ORDER=','.join(map(str,range(route.hypotheses))))
    run_manifest.record_value('auto_route', {
        **{name: os.environ[name] for name in (
            'JETSPEC_ROWS', 'JETSPEC_ROWS_FAST', 'JETSPEC_ROWS_GRAPHS',
            'JETSPEC_F', 'JETSPEC_PERSEQ', 'JETSPEC_FUSE', 'JETSPEC_KSET',
            'JETSPEC_KSET_M', 'JETSPEC_KSET_ORDER')},
        'source': route.source})
    print(f'W252_AUTO N={batch_size} H={route.hypotheses} F=1 J0 P={int(route.per_sequence)} fusion={route.fusion} source={route.source}',flush=True)
    return route
