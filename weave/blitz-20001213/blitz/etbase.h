#ifndef BZ_ETBASE_H
#define BZ_ETBASE_H

BZ_NAMESPACE(blitz)

template<class T>
class ETBase { 
public:
    ETBase() 
    { }

    ETBase(const ETBase<T>&)
    { }
};

BZ_NAMESPACE_END

#endif // BZ_ETBASE_H

