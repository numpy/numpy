/**
 *  Template to represent limbo numbers.
 *
 *  Specializations for integer types that are part of nowhere.
 *  It doesn't support with any real types.
 *
 *  @param Tp Type of the integer. Required to be an integer type.
 *  @param N  Number of elements.
*/
template<typename Tp, std::size_t N>
class DoxyLimbo {
 public:
    /// Default constructor. Initialize nothing.
    DoxyLimbo();
    /// Set Default behavior for copy the limbo.
    DoxyLimbo(const DoxyLimbo<Tp, N> &l);
    /// Returns the raw data for the limbo.
    const Tp *data();
 protected:
    Tp p_data[N]; ///< Example for inline comment.
};
