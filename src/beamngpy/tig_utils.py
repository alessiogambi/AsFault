import logging as log
import traceback

# log = logging.getLogger('beamng.tig_utils')


def close_quietly(o):
    try:
        if o:
            o.close()
    except Exception as ex:
        log.debug('close_quietly exception says:')
        traceback.print_exception(type(ex), ex, ex.__traceback__)


def exec_quietly(command):
    try:
        command()
    except Exception as ex:
        log.debug('exec_quietly exception says:')
        traceback.print_exception(type(ex), ex, ex.__traceback__)
