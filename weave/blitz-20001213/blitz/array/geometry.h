#ifndef BZ_GEOMETRY_H
#define BZ_GEOMETRY_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/geometry.h> must be included after <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

typedef double T_defaultSpatialCoordinate;

template<int N_dim, class T = T_defaultSpatialCoordinate>
class UniformOrthoGeometry {
public:
};

template<int N_dim, class T = T_defaultSpatialCoordinate>
class UniformCubicGeometry {
    T h_;
    T recip_h_;
    T recip2_h_;
    T recip3_h_;
    TinyVector<T,N_dim> zero_;

public:
    typedef T T_coord;

    UniformCubicGeometry()
    {
        h_ = 0.0;
        recip_h_ = 0.0;
        recip2_h_ = 0.0;
        recip3_h_ = 0.0;
        zero_ = 0.0;
    }

    UniformCubicGeometry(T spatialStep)
    {
        h_ = spatialStep;
        zero_ = T(0);
        setup();
    }

    UniformCubicGeometry(T spatialStep, TinyVector<T,N_dim> zeroCoordinates)
    {   
        h_ = spatialStep;
        zero_ = zeroCoordinates;
        setup();
    }    

    TinyVector<T,N_dim> toSpatial(TinyVector<int,N_dim> logicalCoord) const
    {
        return zero_ + h_ * logicalCoord;
    }

    T spatialStep() const
    { return h_; }

    T recipSpatialStep() const
    { return recip_h_; }

    T recipSpatialStepPow2() const
    { return recip2_h_; }

private:
    void setup()
    {
        recip_h_ = 1.0 / h_;
        recip2_h_ = 1.0 / pow2(h_);
        recip3_h_ = 1.0 / pow3(h_);
    }
};

template<int N_dim, class T = T_defaultSpatialCoordinate>
class TensorProductGeometry {
public:
};

BZ_NAMESPACE_END

#endif // BZ_GEOMETRY_H
