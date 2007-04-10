#ifndef _CP_RB_H
#define _CP_RB_H

/** @{ */
/**
 * @file
 *
 * red-black tree definitions. Red-black trees are self balancing binary trees. 
 * red-black trees guarantee a O(log n) time for tree operations. An advantage
 * over AVL trees is in that insertion and deletion require a small number of 
 * rotations (2 or 3) at the most.
 *
 * First introduced by Rudolf Bayer in Symmetric Binary B-Trees: Data 
 * Structures and Maintenance Algorithms, 1972.
 * The 'red-black' terminology is due to Leo J. Guibas and Robert Sedgewick: 
 * A Dichromatic Framework for Balanced Trees, 1978.
 */

#include "common.h"

__BEGIN_DECLS

#include "vector.h"
#include "mempool.h"

struct _cp_rbtree;

#define RB_RED    0
#define RB_BLACK  1

typedef CPROPS_DLL struct _cp_rbnode
{
	void *key;
	void *value;

	/* balance maintainance - color is either 'red' or 'black' */
	int color;

	struct _cp_rbnode *left;
	struct _cp_rbnode *right;
	struct _cp_rbnode *up;
} cp_rbnode;

/* (internal) allocate a new node */
CPROPS_DLL
cp_rbnode *cp_rbnode_create(void *key, void *value, struct _cp_mempool *pool);
/* (internal) deallocate a node */
CPROPS_DLL
void cp_rbtree_destroy_node(struct _cp_rbtree *owner, cp_rbnode *node);
/* (internal) deallocate a node and its subnodes */
CPROPS_DLL
void cp_rbtree_destroy_node_deep(struct _cp_rbtree *owner, cp_rbnode *node);

/* tree wrapper object */
typedef CPROPS_DLL struct _cp_rbtree
{
	cp_rbnode *root;             /* root node */
	
	int items;                   /* item count */

	int mode;					 /* mode flags */
	cp_compare_fn cmp;           /* key comparison function */
	cp_copy_fn key_copy;         /* key copy function */
	cp_destructor_fn key_dtr;    /* key destructor */
	cp_copy_fn value_copy;       /* value copy function */
	cp_destructor_fn value_dtr;  /* value destructor */

	cp_lock *lock;
	cp_thread txowner;           /* set if a transaction is in progress */
	int txtype;                  /* lock type */

	cp_mempool *mempool; 		 /* optional memory pool */
} cp_rbtree;

/* 
 * default create function - equivalent to create_by_option with mode 
 * COLLECTION_MODE_NOSYNC
 */
CPROPS_DLL
cp_rbtree *cp_rbtree_create(cp_compare_fn cmp);
/*
 * complete parameter create function. Note that setting COLLECTION_MODE_COPY
 * without specifying a copy function for either keys or values will result in
 * keys or values respectively being inserted by value, with no copying 
 * performed. Similarly, setting COLLECTION_MODE_DEEP without specifying a 
 * destructor function for keys or values will result in no destructor call
 * for keys or values respectively. This allows using the copy/deep mechanisms
 * for keys only, values only or both.
 */
CPROPS_DLL
cp_rbtree *
	cp_rbtree_create_by_option(int mode, cp_compare_fn cmp, 
							   cp_copy_fn key_copy, cp_destructor_fn key_dtr,
							   cp_copy_fn val_copy, cp_destructor_fn val_dtr);
/* 
 * recursively destroy the tree structure 
 */
CPROPS_DLL
void cp_rbtree_destroy(cp_rbtree *tree);
/*
 * recursively destroy the tree structure with the given destructor functions
 */
CPROPS_DLL
void cp_rbtree_destroy_custom(cp_rbtree *tree, 
							  cp_destructor_fn key_dtr,
							  cp_destructor_fn val_dtr);

/* insertion function */
CPROPS_DLL
void *cp_rbtree_insert(cp_rbtree *tree, void *key, void *value);
/* retrieve the value mapped to the given key */
CPROPS_DLL
void *cp_rbtree_get(cp_rbtree *tree, void *key);
/* return non-zero if a mapping for 'key' could be found */
CPROPS_DLL
int cp_rbtree_contains(cp_rbtree *tree, void *key);
/* delete a mapping */
CPROPS_DLL
void *cp_rbtree_delete(cp_rbtree *tree, void *key);

/* 
 * perform a pre-order iteration over the tree, calling 'callback' on each 
 * node
 */
CPROPS_DLL
int cp_rbtree_callback_preorder(cp_rbtree *tree, 
								cp_callback_fn callback, 
								void *prm);
/* 
 * perform an in-order iteration over the tree, calling 'callback' on each 
 * node
 */
CPROPS_DLL
int cp_rbtree_callback(cp_rbtree *tree, cp_callback_fn callback, void *prm);
/* 
 * perform a post-order iteration over the tree, calling 'callback' on each 
 * node
 */

CPROPS_DLL
int cp_rbtree_callback_postorder(cp_rbtree *tree, 
								 cp_callback_fn callback, 
								 void *prm);

/* return the number of mappings in the tree */
CPROPS_DLL
int cp_rbtree_count(cp_rbtree *tree);

/* 
 * lock tree for reading or writing as specified by type parameter. 
 */
CPROPS_DLL
int cp_rbtree_lock(cp_rbtree *tree, int type);
/* read lock */
#define cp_rbtree_rdlock(tree) (cp_rbtree_lock((tree), COLLECTION_LOCK_READ))
/* write lock */
#define cp_rbtree_wrlock(tree) (cp_rbtree_lock((tree), COLLECTION_LOCK_WRITE))
/* unlock */
CPROPS_DLL
int cp_rbtree_unlock(cp_rbtree *tree);


/* return the table mode indicator */
CPROPS_DLL
int cp_rbtree_get_mode(cp_rbtree *tree);
/* set mode bits on the tree mode indicator */
CPROPS_DLL
int cp_rbtree_set_mode(cp_rbtree *tree, int mode);
/* unset mode bits on the tree mode indicator. if unsetting 
 * COLLECTION_MODE_NOSYNC and the tree was not previously synchronized, the 
 * internal synchronization structure is initalized.
 */
CPROPS_DLL
int cp_rbtree_unset_mode(cp_rbtree *tree, int mode);


/** print tree to stdout */
CPROPS_DLL
void cp_rbtree_dump(cp_rbtree *tree);

/* set tree to use given mempool or allocate a new one if pool is NULL */
CPROPS_DLL
int cp_rbtree_use_mempool(cp_rbtree *tree, cp_mempool *pool);

/* set tree to use a shared memory pool */
CPROPS_DLL
int cp_rbtree_share_mempool(cp_rbtree *tree, struct _cp_shared_mempool *pool);

__END_DECLS

/** @} */

#endif

