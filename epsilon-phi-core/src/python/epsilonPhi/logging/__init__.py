import logging
import logging.config

logging.basicConfig(level=logging.DEBUG)

formatter = logging.Formatter('\x1b[0;30;47m %(asctime)s %(levelname)s %(message)s \x1b[0m')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)

class logMessage(object):
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)
