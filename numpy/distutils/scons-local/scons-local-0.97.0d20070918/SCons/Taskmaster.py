#
# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007 The SCons Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

__doc__ = """
Generic Taskmaster module for the SCons build engine.

This module contains the primary interface(s) between a wrapping user
interface and the SCons build engine.  There are two key classes here:

    Taskmaster
        This is the main engine for walking the dependency graph and
        calling things to decide what does or doesn't need to be built.

    Task
        This is the base class for allowing a wrapping interface to
        decide what does or doesn't actually need to be done.  The
        intention is for a wrapping interface to subclass this as
        appropriate for different types of behavior it may need.

        The canonical example is the SCons native Python interface,
        which has Task subclasses that handle its specific behavior,
        like printing "`foo' is up to date" when a top-level target
        doesn't need to be built, and handling the -c option by removing
        targets as its "build" action.  There is also a separate subclass
        for suppressing this output when the -q option is used.

        The Taskmaster instantiates a Task object for each (set of)
        target(s) that it decides need to be evaluated and/or built.
"""

__revision__ = "src/engine/SCons/Taskmaster.py 2446 2007/09/18 11:41:57 knight"

import SCons.compat

import operator
import string
import sys
import traceback

import SCons.Node
import SCons.Errors

StateString = SCons.Node.StateString



# A subsystem for recording stats about how different Nodes are handled by
# the main Taskmaster loop.  There's no external control here (no need for
# a --debug= option); enable it by changing the value of CollectStats.

CollectStats = None

class Stats:
    """
    A simple class for holding statistics about the disposition of a
    Node by the Taskmaster.  If we're collecting statistics, each Node
    processed by the Taskmaster gets one of these attached, in which case
    the Taskmaster records its decision each time it processes the Node.
    (Ideally, that's just once per Node.)
    """
    def __init__(self):
        """
        Instantiates a Taskmaster.Stats object, initializing all
        appropriate counters to zero.
        """
        self.considered  = 0
        self.already_handled  = 0
        self.problem  = 0
        self.child_failed  = 0
        self.not_built  = 0
        self.side_effects  = 0
        self.build  = 0

StatsNodes = []

fmt = "%(considered)3d "\
      "%(already_handled)3d " \
      "%(problem)3d " \
      "%(child_failed)3d " \
      "%(not_built)3d " \
      "%(side_effects)3d " \
      "%(build)3d "

def dump_stats():
    StatsNodes.sort(lambda a, b: cmp(str(a), str(b)))
    for n in StatsNodes:
        print (fmt % n.stats.__dict__) + str(n)



class Task:
    """
    Default SCons build engine task.

    This controls the interaction of the actual building of node
    and the rest of the engine.

    This is expected to handle all of the normally-customizable
    aspects of controlling a build, so any given application
    *should* be able to do what it wants by sub-classing this
    class and overriding methods as appropriate.  If an application
    needs to customze something by sub-classing Taskmaster (or
    some other build engine class), we should first try to migrate
    that functionality into this class.

    Note that it's generally a good idea for sub-classes to call
    these methods explicitly to update state, etc., rather than
    roll their own interaction with Taskmaster from scratch.
    """
    def __init__(self, tm, targets, top, node):
        self.tm = tm
        self.targets = targets
        self.top = top
        self.node = node
        self.exc_clear()

    def display(self, message):
        """
        Hook to allow the calling interface to display a message.

        This hook gets called as part of preparing a task for execution
        (that is, a Node to be built).  As part of figuring out what Node
        should be built next, the actually target list may be altered,
        along with a message describing the alteration.  The calling
        interface can subclass Task and provide a concrete implementation
        of this method to see those messages.
        """
        pass

    def prepare(self):
        """
        Called just before the task is executed.

        This is mainly intended to give the target Nodes a chance to
        unlink underlying files and make all necessary directories before
        the Action is actually called to build the targets.
        """

        # Now that it's the appropriate time, give the TaskMaster a
        # chance to raise any exceptions it encountered while preparing
        # this task.
        self.exception_raise()

        if self.tm.message:
            self.display(self.tm.message)
            self.tm.message = None

        for t in self.targets:
            t.prepare()
            for s in t.side_effects:
                s.prepare()

    def get_target(self):
        """Fetch the target being built or updated by this task.
        """
        return self.node

    def execute(self):
        """
        Called to execute the task.

        This method is called from multiple threads in a parallel build,
        so only do thread safe stuff here.  Do thread unsafe stuff in
        prepare(), executed() or failed().
        """

        try:
            everything_was_cached = 1
            for t in self.targets:
                if not t.retrieve_from_cache():
                    everything_was_cached = 0
                    break
            if not everything_was_cached:
                self.targets[0].build()
        except KeyboardInterrupt:
            raise
        except SystemExit:
            exc_value = sys.exc_info()[1]
            raise SCons.Errors.ExplicitExit(self.targets[0], exc_value.code)
        except SCons.Errors.UserError:
            raise
        except SCons.Errors.BuildError:
            raise
        except:
            raise SCons.Errors.TaskmasterException(self.targets[0],
                                                   sys.exc_info())

    def executed_without_callbacks(self):
        """
        Called when the task has been successfully executed
        and the Taskmaster instance doesn't want to call
        the Node's callback methods.
        """
        for t in self.targets:
            if t.get_state() == SCons.Node.executing:
                for side_effect in t.side_effects:
                    side_effect.set_state(SCons.Node.no_state)
                t.set_state(SCons.Node.executed)

    def executed_with_callbacks(self):
        """
        Called when the task has been successfully executed and
        the Taskmaster instance wants to call the Node's callback
        methods.

        This may have been a do-nothing operation (to preserve build
        order), so we must check the node's state before deciding whether
        it was "built", in which case we call the appropriate Node method.
        In any event, we always call "visited()", which will handle any
        post-visit actions that must take place regardless of whether
        or not the target was an actual built target or a source Node.
        """
        for t in self.targets:
            if t.get_state() == SCons.Node.executing:
                for side_effect in t.side_effects:
                    side_effect.set_state(SCons.Node.no_state)
                t.set_state(SCons.Node.executed)
                t.built()
            t.visited()

    executed = executed_with_callbacks

    def failed(self):
        """
        Default action when a task fails:  stop the build.
        """
        self.fail_stop()

    def fail_stop(self):
        """
        Explicit stop-the-build failure.
        """
        for t in self.targets:
            t.set_state(SCons.Node.failed)
        self.tm.stop()

        # We're stopping because of a build failure, but give the
        # calling Task class a chance to postprocess() the top-level
        # target under which the build failure occurred.
        self.targets = [self.tm.current_top]
        self.top = 1

    def fail_continue(self):
        """
        Explicit continue-the-build failure.

        This sets failure status on the target nodes and all of
        their dependent parent nodes.
        """
        for t in self.targets:
            # Set failure state on all of the parents that were dependent
            # on this failed build.
            def set_state(node): node.set_state(SCons.Node.failed)
            t.call_for_all_waiting_parents(set_state)

    def make_ready_all(self):
        """
        Marks all targets in a task ready for execution.

        This is used when the interface needs every target Node to be
        visited--the canonical example being the "scons -c" option.
        """
        self.out_of_date = self.targets[:]
        for t in self.targets:
            t.disambiguate().set_state(SCons.Node.executing)
            for s in t.side_effects:
                s.set_state(SCons.Node.executing)

    def make_ready_current(self):
        """
        Marks all targets in a task ready for execution if any target
        is not current.

        This is the default behavior for building only what's necessary.
        """
        self.out_of_date = []
        for t in self.targets:
            try:
                t.disambiguate().make_ready()
                is_up_to_date = not t.has_builder() or \
                                (not t.always_build and t.is_up_to_date())
            except EnvironmentError, e:
                raise SCons.Errors.BuildError(node=t, errstr=e.strerror, filename=e.filename)
            if is_up_to_date:
                t.set_state(SCons.Node.up_to_date)
            else:
                self.out_of_date.append(t)
                t.set_state(SCons.Node.executing)
                for s in t.side_effects:
                    s.set_state(SCons.Node.executing)

    make_ready = make_ready_current

    def postprocess(self):
        """
        Post-processes a task after it's been executed.

        This examines all the targets just built (or not, we don't care
        if the build was successful, or even if there was no build
        because everything was up-to-date) to see if they have any
        waiting parent Nodes, or Nodes waiting on a common side effect,
        that can be put back on the candidates list.
        """

        # We may have built multiple targets, some of which may have
        # common parents waiting for this build.  Count up how many
        # targets each parent was waiting for so we can subtract the
        # values later, and so we *don't* put waiting side-effect Nodes
        # back on the candidates list if the Node is also a waiting
        # parent.

        targets = set(self.targets)

        parents = {}
        for t in targets:
            for p in t.waiting_parents.keys():
                parents[p] = parents.get(p, 0) + 1

        for t in targets:
            for s in t.side_effects:
                if s.get_state() == SCons.Node.executing:
                    s.set_state(SCons.Node.no_state)
                    for p in s.waiting_parents.keys():
                        if not parents.has_key(p):
                            parents[p] = 1
                for p in s.waiting_s_e.keys():
                    if p.ref_count == 0:
                        self.tm.candidates.append(p)

        for p, subtract in parents.items():
            p.ref_count = p.ref_count - subtract
            if p.ref_count == 0:
                self.tm.candidates.append(p)

        for t in targets:
            t.postprocess()

    # Exception handling subsystem.
    #
    # Exceptions that occur while walking the DAG or examining Nodes
    # must be raised, but must be raised at an appropriate time and in
    # a controlled manner so we can, if necessary, recover gracefully,
    # possibly write out signature information for Nodes we've updated,
    # etc.  This is done by having the Taskmaster tell us about the
    # exception, and letting

    def exc_info(self):
        """
        Returns info about a recorded exception.
        """
        return self.exception

    def exc_clear(self):
        """
        Clears any recorded exception.

        This also changes the "exception_raise" attribute to point
        to the appropriate do-nothing method.
        """
        self.exception = (None, None, None)
        self.exception_raise = self._no_exception_to_raise

    def exception_set(self, exception=None):
        """
        Records an exception to be raised at the appropriate time.

        This also changes the "exception_raise" attribute to point
        to the method that will, in fact
        """
        if not exception:
            exception = sys.exc_info()
        self.exception = exception
        self.exception_raise = self._exception_raise

    def _no_exception_to_raise(self):
        pass

    def _exception_raise(self):
        """
        Raises a pending exception that was recorded while getting a
        Task ready for execution.
        """
        exc = self.exc_info()[:]
        try:
            exc_type, exc_value, exc_traceback = exc
        except ValueError:
            exc_type, exc_value = exc
            exc_traceback = None
        raise exc_type, exc_value, exc_traceback


def find_cycle(stack):
    if stack[0] == stack[-1]:
        return stack
    for n in stack[-1].waiting_parents.keys():
        stack.append(n)
        if find_cycle(stack):
            return stack
        stack.pop()
    return None


class Taskmaster:
    """
    The Taskmaster for walking the dependency DAG.
    """

    def __init__(self, targets=[], tasker=Task, order=None, trace=None):
        self.original_top = targets
        self.top_targets_left = targets[:]
        self.top_targets_left.reverse()
        self.candidates = []
        self.tasker = tasker
        if not order:
            order = lambda l: l
        self.order = order
        self.message = None
        self.trace = trace
        self.next_candidate = self.find_next_candidate

    def find_next_candidate(self):
        """
        Returns the next candidate Node for (potential) evaluation.

        The candidate list (really a stack) initially consists of all of
        the top-level (command line) targets provided when the Taskmaster
        was initialized.  While we walk the DAG, visiting Nodes, all the
        children that haven't finished processing get pushed on to the
        candidate list.  Each child can then be popped and examined in
        turn for whether *their* children are all up-to-date, in which
        case a Task will be created for their actual evaluation and
        potential building.

        Here is where we also allow candidate Nodes to alter the list of
        Nodes that should be examined.  This is used, for example, when
        invoking SCons in a source directory.  A source directory Node can
        return its corresponding build directory Node, essentially saying,
        "Hey, you really need to build this thing over here instead."
        """
        try:
            return self.candidates.pop()
        except IndexError:
            pass
        try:
            node = self.top_targets_left.pop()
        except IndexError:
            return None
        self.current_top = node
        alt, message = node.alter_targets()
        if alt:
            self.message = message
            self.candidates.append(node)
            self.candidates.extend(self.order(alt))
            node = self.candidates.pop()
        return node

    def no_next_candidate(self):
        """
        Stops Taskmaster processing by not returning a next candidate.
        """
        return None

    def _find_next_ready_node(self):
        """
        Finds the next node that is ready to be built.

        This is *the* main guts of the DAG walk.  We loop through the
        list of candidates, looking for something that has no un-built
        children (i.e., that is a leaf Node or has dependencies that are
        all leaf Nodes or up-to-date).  Candidate Nodes are re-scanned
        (both the target Node itself and its sources, which are always
        scanned in the context of a given target) to discover implicit
        dependencies.  A Node that must wait for some children to be
        built will be put back on the candidates list after the children
        have finished building.  A Node that has been put back on the
        candidates list in this way may have itself (or its sources)
        re-scanned, in order to handle generated header files (e.g.) and
        the implicit dependencies therein.

        Note that this method does not do any signature calculation or
        up-to-date check itself.  All of that is handled by the Task
        class.  This is purely concerned with the dependency graph walk.
        """

        self.ready_exc = None

        T = self.trace

        while 1:
            node = self.next_candidate()
            if node is None:
                return None

            node = node.disambiguate()
            state = node.get_state()

            if CollectStats:
                if not hasattr(node, 'stats'):
                    node.stats = Stats()
                    StatsNodes.append(node)
                S = node.stats
                S.considered = S.considered + 1
            else:
                S = None

            if T: T.write('Taskmaster: %s:' % repr(str(node)))

            # Skip this node if it has already been evaluated:
            if state > SCons.Node.pending:
                if S: S.already_handled = S.already_handled + 1
                if T: T.write(' already handled (%s)\n' % StateString[state])
                continue

            # Mark this node as being on the execution stack:
            node.set_state(SCons.Node.pending)

            try:
                children = node.children()
            except SystemExit:
                exc_value = sys.exc_info()[1]
                e = SCons.Errors.ExplicitExit(node, exc_value.code)
                self.ready_exc = (SCons.Errors.ExplicitExit, e)
                if T: T.write(' SystemExit\n')
                return node
            except KeyboardInterrupt:
                if T: T.write(' KeyboardInterrupt\n')
                raise
            except:
                # We had a problem just trying to figure out the
                # children (like a child couldn't be linked in to a
                # BuildDir, or a Scanner threw something).  Arrange to
                # raise the exception when the Task is "executed."
                self.ready_exc = sys.exc_info()
                if S: S.problem = S.problem + 1
                if T: T.write(' exception\n')
                return node

            if T and children:
                c = map(str, children)
                c.sort()
                T.write(' children:\n    %s\n   ' % c)

            childstate = map(lambda N: (N, N.get_state()), children)

            # Skip this node if any of its children have failed.  This
            # catches the case where we're descending a top-level target
            # and one of our children failed while trying to be built
            # by a *previous* descent of an earlier top-level target.
            failed_children = filter(lambda I: I[1] == SCons.Node.failed,
                                     childstate)
            if failed_children:
                node.set_state(SCons.Node.failed)
                if S: S.child_failed = S.child_failed + 1
                if T:
                    c = map(str, failed_children)
                    c.sort()
                    T.write(' children failed:\n    %s\n' % c)
                continue

            # Detect dependency cycles:
            pending_nodes = filter(lambda I: I[1] == SCons.Node.pending, childstate)
            if pending_nodes:
                for p in pending_nodes:
                    cycle = find_cycle([p[0], node])
                    if cycle:
                        desc = "Dependency cycle: " + string.join(map(str, cycle), " -> ")
                        if T: T.write(' dependency cycle\n')
                        raise SCons.Errors.UserError, desc

            not_built = filter(lambda I: I[1] <= SCons.Node.executing, childstate)
            if not_built:
                # We're waiting on one or more derived targets that have
                # not yet finished building.

                not_visited = filter(lambda I: not I[1], not_built)
                if not_visited:
                    # Some of them haven't even been visited yet.
                    # Add them to the list so that on some next pass
                    # we can take a stab at evaluating them (or
                    # their children).
                    not_visited = map(lambda I: I[0], not_visited)
                    not_visited.reverse()
                    self.candidates.extend(self.order(not_visited))

                n_b_nodes = map(lambda I: I[0], not_built)

                # Add this node to the waiting parents lists of anything
                # we're waiting on, with a reference count so we can be
                # put back on the list for re-evaluation when they've
                # all finished.
                map(lambda n, P=node: n.add_to_waiting_parents(P), n_b_nodes)
                node.ref_count = len(set(n_b_nodes))

                if S: S.not_built = S.not_built + 1
                if T:
                    c = map(str, n_b_nodes)
                    c.sort()
                    T.write(' waiting on unfinished children:\n    %s\n' % c)
                continue

            # Skip this node if it has side-effects that are
            # currently being built:
            side_effects = filter(lambda N:
                                  N.get_state() == SCons.Node.executing,
                                  node.side_effects)
            if side_effects:
                map(lambda n, P=node: n.add_to_waiting_s_e(P), side_effects)
                if S: S.side_effects = S.side_effects + 1
                if T:
                    c = map(str, side_effects)
                    c.sort()
                    T.write(' waiting on side effects:\n    %s\n' % c)
                continue

            # The default when we've gotten through all of the checks above:
            # this node is ready to be built.
            if S: S.build = S.build + 1
            if T: T.write(' evaluating %s\n' % node)
            return node

        return None

    def next_task(self):
        """
        Returns the next task to be executed.

        This simply asks for the next Node to be evaluated, and then wraps
        it in the specific Task subclass with which we were initialized.
        """
        node = self._find_next_ready_node()

        if node is None:
            return None

        tlist = node.get_executor().targets

        task = self.tasker(self, tlist, node in self.original_top, node)
        try:
            task.make_ready()
        except KeyboardInterrupt:
            raise
        except:
            # We had a problem just trying to get this task ready (like
            # a child couldn't be linked in to a BuildDir when deciding
            # whether this node is current).  Arrange to raise the
            # exception when the Task is "executed."
            self.ready_exc = sys.exc_info()

        if self.ready_exc:
            task.exception_set(self.ready_exc)

        self.ready_exc = None

        return task

    def stop(self):
        """
        Stops the current build completely.
        """
        self.next_candidate = self.no_next_candidate
