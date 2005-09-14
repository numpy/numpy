#
# Title: Provides ParallelExec to execute commands in
#        other (background or parallel) threads.
# Author: Pearu Peteson <pearu@cens.ioc.ee>
# Created: October, 2003
#

__all__ = ['ParallelExec']

import sys
import threading
import Queue
import traceback
import types
import inspect
import time
import atexit

class ParallelExec(threading.Thread):
    """ Create a thread of parallel execution.
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.__queue = Queue.Queue(0)
        self.__frame = sys._getframe(1)
        self.setDaemon(1)
        self.start()

    def __call__(self,code,frame=None,wait=0):
        """ Execute code in parallel thread inside given frame (default
        frame is where this instance was created).
        If wait is True then __call__ returns after code is executed,
        otherwise code execution happens in background.
        """
        if wait:
            wait_for_code = threading.Event()
        else:
            wait_for_code = None
        self.__queue.put((code,frame,wait_for_code))
        if wait:
            wait_for_code.wait()

    def shutdown(self):
        """ Shutdown parallel thread."""
        self.__queue.put((None,None,None))

    def run(self):
        """ Called by threading.Thread."""
        while 1:
            code, frame, wait_for_code = self.__queue.get()
            if code is None:
                break
            if frame is None:
                frame = self.__frame
            try:
                exec (code, frame.f_globals,frame.f_locals)
            except Exception:
                try:
                    traceback.print_exc()
                except AttributeError:
                    pass
            if wait_for_code is not None:
                wait_for_code.set()
