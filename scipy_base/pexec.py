#
# Title: Provides ParallelExec to execute commands in
#        other (background or parallel) threads.
# Author: Pearu Peteson <pearu@cens.ioc.ee>
# Created: October, 2003
#

__all__ = ['ParallelExec','migrate']

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
        self.__queue.put((None,None))

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
                traceback.print_exc()
            if wait_for_code is not None:
                wait_for_code.set()

def migrate(obj, caller):
    """ Return obj wrapper that facilitates accessing object
    from another thread."""
    if inspect.isroutine(obj):
        return MigratedRoutine(obj, caller)
    raise NotImplementedError,`type(obj)`

class Attrs:
    def __init__(self,**kws):
        for k,v in kws.items():
            setattr(self,k,v)

class MigratedRoutine:
    """ Wrapper for calling routines from another thread.

    func   - function or built-in or method
    caller('<command>',<frame>) - executes command in another thread
    """
    def __init__(self, func, caller):
        self.__attrs = Attrs(func=func, caller=caller, finished=threading.Event())
        for n,v in inspect.getmembers(func):
            if n in ['__dict__','__class__','__call__','__attrs']:
                continue
            setattr(self,n,v)

    def __call__(self, *args, **kws):
        attrs = self.__attrs
        frame = sys._getframe(0)
        attrs.finished.clear()
        attrs.caller('attrs.result = attrs.func(*args, **kws)',frame)
        attrs.caller('attrs.finished.set()',frame)
        attrs.finished.wait()
        result = attrs.result
        attrs.result = None
        return result
