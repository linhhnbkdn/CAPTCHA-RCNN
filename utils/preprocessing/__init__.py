import sys
import os
sys.path.insert(0, os.getcwd())

import logging
from utils import logger

LOGGER = logging.getLogger('preprocessing')

from .loader import Loader
from .preprocessing import Preprocessing