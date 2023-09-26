from openood.utils import Config

from .ad_recorder import ADRecorder
from .arpl_recorder import ARPLRecorder
from .base_recorder import BaseRecorder
from .cutpaste_recorder import CutpasteRecorder
from .draem_recorder import DRAEMRecorder
from .dsvdd_recorder import DCAERecorder, DSVDDRecorder
from .kdad_recorder import KdadRecorder
from .opengan_recorder import OpenGanRecorder
from .rd4ad_recorder import Rd4adRecorder
from .mwe_recorder import MWERecorder

def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'mwe': MWERecorder,
        'draem': DRAEMRecorder,
        'opengan': OpenGanRecorder,
        'dcae': DCAERecorder,
        'dsvdd': DSVDDRecorder,
        'kdad': KdadRecorder,
        'arpl': ARPLRecorder,
        'cutpaste': CutpasteRecorder,
        'ad': ADRecorder,
        'rd4ad': Rd4adRecorder,
    }

    return recorders[config.recorder.name](config)
