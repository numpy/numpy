# Colored log, requires Python 2.3 or up.

import sys
from distutils.log import *
from distutils.log import Log as old_Log
from distutils.log import _global_log
from misc_util import red_text, yellow_text, cyan_text, green_text, is_sequence, is_string


def _fix_args(args,flag=1):
    if is_string(args):
        return args.replace('%','%%')
    if flag and is_sequence(args):
        return tuple([_fix_args(a,flag=0) for a in args])
    return args

class Log(old_Log):
    def _log(self, level, msg, args):
        if level >= self.threshold:
            if args:
                print _global_color_map[level](msg % _fix_args(args))
            else:
                print _global_color_map[level](msg)
            sys.stdout.flush()

    def good(self, msg, *args):
        """If we'd log WARN messages, log this message as a 'nice' anti-warn
        message.
        """
        if WARN >= self.threshold:
            if args:
                print green_text(msg % _fix_args(args))
            else:
                print green_text(msg)
            sys.stdout.flush()
_global_log.__class__ = Log

good = _global_log.good

def set_threshold(level):
    prev_level = _global_log.threshold
    if prev_level > DEBUG:
        # If we're running at DEBUG, don't change the threshold, as there's
        # likely a good reason why we're running at this level.
        _global_log.threshold = level
    return prev_level

def set_verbosity(v):
    prev_level = _global_log.threshold
    if v < 0:
        set_threshold(ERROR)
    elif v == 0:
        set_threshold(WARN)
    elif v == 1:
        set_threshold(INFO)
    elif v >= 2:
        set_threshold(DEBUG)
    return {FATAL:-2,ERROR:-1,WARN:0,INFO:1,DEBUG:2}.get(prev_level,1)

_global_color_map = {
    DEBUG:cyan_text,
    INFO:yellow_text,
    WARN:red_text,
    ERROR:red_text,
    FATAL:red_text
}

set_verbosity(INFO)
