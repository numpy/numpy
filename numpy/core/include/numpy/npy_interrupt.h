
/* Signal handling: 

In your C-extension:  

Around a block of code you want to be interruptable 

NPY_SIG_ON
[code]
NPY_SIG_OFF

*/

/* Add signal handling macros */

#ifndef NPY_INTERRUPT_H
#define NPY_INTERRUPT_H

#define NPY_SIG_ON
#define NPY_SIG_OFF
#define NPY_SIG_CHECK


#endif /* NPY_INTERRUPT_H */
