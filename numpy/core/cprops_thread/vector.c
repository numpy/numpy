/**
 * cp_vector is a 'safe array' implementation
 */

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "log.h"
#include "common.h"

#include "vector.h"

cp_vector *cp_vector_create_by_option(int size, 
									  int mode, 
									  cp_copy_fn copy_item,
									  cp_destructor_fn free_item)
{
	cp_vector *v = calloc(1, sizeof(cp_vector));
	if (v == NULL) 
	{
		errno = ENOMEM;
		return NULL;
	}

	v->mem = calloc(size, sizeof(void *));
	if (v->mem == NULL)
	{
		errno = ENOMEM;
		return NULL;
	}

	v->size = size;
	v->mode = mode;
	v->copy_item = copy_item;
	v->free_item = free_item;
	v->head = v->tail = 0;

	return v;
}

cp_vector *cp_vector_create(int size)
{
	return cp_vector_create_by_option(size, 0, NULL, NULL);
}

cp_vector *cp_vector_wrap(void **data, int len, int mode)
{
	cp_vector *v = calloc(1, sizeof(cp_vector));
	if (v == NULL) return NULL;

	v->mem = data;
	v->size = len;
	v->mode = mode;
	v->head = len;
	v->tail = 0;

	return v;
}

void cp_vector_destroy(cp_vector *v)
{
	if (v)
	{
		if ((v->mode & COLLECTION_MODE_DEEP) && v->free_item != NULL)
		{
			int i;
			int n = cp_vector_size(v);
			void *item;
			for (i = 0; i < n; i++)
			{
				item = cp_vector_element_at(v, i);
				if (item) (*v->free_item)(item);
			}
		}
		free(v->mem);
		free(v);
	}
}

void cp_vector_destroy_custom(cp_vector *v, cp_destructor_fn dtr)
{
	if (v)
	{
		int i;
		int n = cp_vector_size(v);

		if (dtr)
		{
			for (i = 0; i < n; i++)
				if (v->mem[i]) (*dtr)(v->mem[i]);
		}
	
		free(v->mem);
		free(v);
	}
}

void *cp_vector_element_at(cp_vector *v, int index)
{
	return index >= 0 && index <= cp_vector_size(v) ? v->mem[index] : NULL;
}

void *cp_vector_set_element(cp_vector *v, int index, void *element)
{
	if (index < 0)
	{
		errno = EINVAL;
		return NULL;
	}

	if (index >= v->head)
		v->head = index + 1;

	if (v->head >= v->size)
	{
		v->size = index + 2;
		v->mem = realloc(v->mem, v->size * sizeof(void *));
	}

	if ((v->mode & COLLECTION_MODE_DEEP) && 
			v->mem[index] != NULL && v->free_item != NULL)
		(*v->free_item)(v->mem[index]);

	if ((v->mode & COLLECTION_MODE_COPY) && element != NULL && v->copy_item != NULL)
		v->mem[index] = (*v->copy_item)(element);
	else
		v->mem[index] = element;
	
	return element;
}

void *cp_vector_add_element(cp_vector *v, void *element)
{
	void *addr = NULL;

	if (v->head + 1 >= v->tail + v->size)
	{
		void **newptr = realloc(v->mem, 2 * v->size * sizeof(void *));
		if (newptr == NULL) return NULL;
		v->mem = newptr;
		if (v->head < v->tail)
		{
			memcpy(v->mem, &v->mem[v->size], v->head * sizeof(void *));
			v->head += v->size;
		}
		v->size *= 2;
	}
			
	if (v->mode & COLLECTION_MODE_COPY)
		v->mem[v->head] = (*v->copy_item)(element);
	else
		v->mem[v->head] = element;
	addr = v->mem[v->head];
	v->head = (v->head + 1) % v->size;

	return addr;
}

int cp_vector_size(cp_vector *v)
{
	return (v->head - v->tail + v->size) % v->size;
}

