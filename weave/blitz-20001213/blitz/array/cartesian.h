#ifndef BZ_ARRAY_CARTESIAN_H
#define BZ_ARRAY_CARTESIAN_H

BZ_NAMESPACE(blitz)

/*
 * CartesianProduct<T_tuple,T_container> is an adaptor which represents
 * the cartesian product of several containers.  
 */

// forward declaration of iterator
template<class T_tuple, class T_container, int N_containers>
class CartesianProductIterator;

struct _cp_end_tag { };

template<class T_tuple, class T_container, int N_containers>
class CartesianProduct {
public:
    typedef T_tuple value_type;
    typedef T_tuple& reference;
    typedef const T_tuple& const_reference;
    typedef CartesianProductIterator<T_tuple,T_container,N_containers> iterator;
    typedef int difference_type;
    typedef int size_type;

    iterator begin()
    { return iterator(*this); }

    iterator end()
    { return iterator(_cp_end_tag()); }

    CartesianProduct(const T_container& container0, 
        const T_container& container1)
    { 
        BZPRECONDITION(N_containers == 2);
        containers_[0] = &container0;
        containers_[1] = &container1;
    }

    CartesianProduct(const T_container& container0, 
        const T_container& container1,
        const T_container& container2)
    { 
        BZPRECONDITION(N_containers == 3);
        containers_[0] = &container0;
        containers_[1] = &container1;
        containers_[2] = &container2;
    }

    const T_container& operator[](int i)
    { return *(containers_[i]); }

    void debugDump();

protected:
    const T_container* containers_[N_containers]; 
};

template<class T_tuple, class T_container, int N_containers>
void CartesianProduct<T_tuple,T_container,N_containers>::debugDump()
{
    cout << "Dump of CartesianProduct<..,..," << N_containers << ">" << endl;
    for (int i=0; i < N_containers; ++i)
    {
        cout << "Container " << (i+1) << ": ";
        _bz_typename T_container::const_iterator iter = containers_[i]->begin(),
            end = containers_[i]->end();
        for (; iter != end; ++iter)
            cout << (*iter) << '\t'; 
    }
}

template<class T_tuple, class T_container, int N_containers>
class CartesianProductIterator {
public:
    typedef _bz_typename T_container::const_iterator citerator;
    typedef CartesianProductIterator<T_tuple,T_container,N_containers> iterator;
    typedef CartesianProduct<T_tuple,T_container,N_containers> T_cp;

    CartesianProductIterator(T_cp& container)
    {
        for (int i=0; i < N_containers; ++i)
        {
            firstiters_[i] = container[i].begin();
            iters_[i] = firstiters_[i];
            enditers_[i] = container[i].end();
            tuple_[i] = *iters_[i];
        }

        endflag_ = _bz_false;
    }

    void operator++();

    CartesianProductIterator(_cp_end_tag)
    {
        endflag_ = _bz_true;
    }

    _bz_bool operator==(const iterator& x) const
    {
        return (endflag_ == x.endflag_);
    }

    _bz_bool operator!=(const iterator& x) const
    {   
        return endflag_ != x.endflag_;
    }

    const T_tuple& operator*() const
    { return tuple_; }

protected:
    citerator iters_[N_containers];
    citerator firstiters_[N_containers];
    citerator enditers_[N_containers];
    T_tuple tuple_;
    _bz_bool endflag_;
};

template<class T_tuple, class T_container, int N_containers>
void CartesianProductIterator<T_tuple, T_container, 
    N_containers>::operator++()
{
    // NEEDS_WORK: put in short-circuit for most common case
    // (just increment the last iterator)

    // Usual stack-style increment
    const int Nminus1 = N_containers - 1;

    int i = Nminus1;

    for (; i >= 0; --i)
    {
        ++iters_[i];
        if (iters_[i] != enditers_[i])
            break;
    }

    if (i == -1)
    {
        endflag_ = _bz_true;
        return;
    }

    tuple_[i] = *iters_[i];

    for (++i; i < N_containers; ++i)  
    {
        iters_[i] = firstiters_[i];
        tuple_[i] = *iters_[i];
    }
}

BZ_NAMESPACE_END

#endif // BZ_ARRAY_CARTESIAN_H

