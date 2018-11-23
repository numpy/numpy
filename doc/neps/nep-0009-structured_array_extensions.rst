===================================
NEP 9 — Structured array extensions
===================================

:Status: Deferred

1.  Create with-style context that makes "named-columns" available as names in the namespace.

   with np.columns(array):
        price = unit * quantityt


2. Allow structured arrays to be sliced by their column  (i.e. one additional indexing option for structured arrays) so that a[:4, 'foo':'bar']  would be allowed.
