#ifndef BZ_DOMAIN_H
#define BZ_DOMAIN_H

#ifndef BZ_TINYVEC_H
 #include <blitz/tinyvec.h>
#endif

#ifndef BZ_RANGE_H
 #include <blitz/range.h>
#endif

/*
 * Portions of this class were inspired by the "RectDomain" class
 * provided by the Titanium language (UC Berkeley).
 */

BZ_NAMESPACE(blitz)

template<int N_rank>
class RectDomain {

public:
    RectDomain(const TinyVector<int,N_rank>& lbound,
        const TinyVector<int,N_rank>& ubound)
      : lbound_(lbound), ubound_(ubound)
    { }

    // NEEDS_WORK: better constructors
    // RectDomain(Range, Range, ...)
    // RectDomain with any combination of Range and int

    const TinyVector<int,N_rank>& lbound() const
    { return lbound_; }

    int lbound(int i) const
    { return lbound_(i); }

    const TinyVector<int,N_rank>& ubound() const
    { return ubound_; }

    int ubound(int i) const
    { return ubound_(i); }

    Range operator[](int rank) const
    { return Range(lbound_(rank), ubound_(rank)); }

    void shrink(int amount)
    {
        lbound_ += amount;
        ubound_ -= amount;
    }

    void shrink(int dim, int amount)
    {
        lbound_(dim) += amount;
        ubound_(dim) -= amount;
    }

    void expand(int amount)
    {
        lbound_ -= amount;
        ubound_ += amount;
    }

    void expand(int dim, int amount)
    {
        lbound_(dim) -= amount;
        ubound_(dim) += amount;
    }

private:
    TinyVector<int,N_rank> lbound_, ubound_;
};

template<int N_rank>
inline RectDomain<N_rank> strip(const TinyVector<int,N_rank>& startPosition,
    int stripDimension, int ubound)
{
    BZPRECONDITION((stripDimension >= 0) && (stripDimension < N_rank));
    BZPRECONDITION(ubound >= startPosition(stripDimension));

    TinyVector<int,N_rank> endPosition = startPosition;
    endPosition(stripDimension) = ubound;
    return RectDomain<N_rank>(startPosition, endPosition);
}

BZ_NAMESPACE_END

#endif // BZ_DOMAIN_H
