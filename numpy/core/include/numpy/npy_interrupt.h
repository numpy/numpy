
/* Signal handling: 

This header file defines macros that allow your code to handle
interrupts received during processing.  Interrupts that 
could reasonably be handled:

SIGINT, SIGABRT, SIGALRM, SIGSEGV

****Warning***************

Do not allow code that creates temporary memory or increases reference
counts of Python objects to be interrupted unless you handle decrementing
the reference counts and freeing any allocated memory in the clean-up code.

**************************

The mechanism for handling interrupts is conceptually simple:

  - replace the signal handler with our own home-grown version
     and store the old one.  
  - run the code to be interrupted -- if an interrupt occurs
     the handler should basically just cause a return to the
     calling function for clean-up work. 
  - restore the old signal handler 

Of course, every code that allows interrupts must account for
returning via the interrupt and handle clean-up correctly.  But,
even still, the simple paradigm is complicated by at least three
factors.

 1) platform portability (i.e. Microsoft says not to use longjmp
     to return from signal handling.  They have a __try  and __except 
     extension to C instead but what about mingw?).
 2) how to handle threads
     a) apparently whether signals are delivered to every thread of
        the process or the "invoking" thread is platform dependent. 
     b) if we use global variables to save state, then how is this
        to be done in a thread-safe way.
 3) A general-purpose facility must allow for the possibility of
    re-entrance (i.e. during execution of the code that is allowed
    to interrupt, we might call back into this very section of code
    serially). 

Ideas:

 1) Start by implementing an approach that works on platforms that
    can use setjmp and longjmp functionality and does nothing 
    on other platforms.  Initially only catch SIGINT.

 2) Handle threads by storing global information in a linked-list
    with a process-id key.  Then use a call-back function that longjmps
    only to the correct buffer.

 3) Store a local copy of the global information and restore it on clean-up
    so that re-entrance works. 


Interface:

In your C-extension.  around a block of code you want to be interruptable 

NPY_SIG_TRY {
[code]
}
NPY_SIG_EXCEPT(sigval) {  
[signal return]
}
NPY_SIG_ELSE 
[normal return]

sigval is a local variable that will receive what
signal was received.  You can use it to perform different
actions based on the signal received. 

Default actions (setting of specific Python errors)
can be obtained with

NPY_SIG_TRY {
[code]
NPY_SIG_EXCEPT_GOTO(label)
[normal return]

label:
  [error return]
*/

/* Add signal handling macros */

#ifndef NPY_INTERRUPT_H
#define NPY_INTERRUPT_H

#ifdef NPY_NO_SIGNAL

#define NPY_SIG_ON
#define NPY_SIG_OFF

#else

#define NPY_SIG_ON
#define NPY_SIG_OFF

#endif /* NPY_NO_SIGNAL */

#endif /* NPY_INTERRUPT_H */
