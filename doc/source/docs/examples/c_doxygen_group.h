///@addtogroup C_doxygen_group_example
///@{

/// An example of documented data type.
typedef struct {
    int size; ///< Size of allacoted data.
    void *ptr; ///< Pointer of allacoted data.
} cdoxy_data1;

enum cdoxy_nowhere_in {
    cdoxy_earth = 1, ///< A reigon of the earth
    cdoxy_moon, ///< A reigon of the moon
    cdoxy_sun ///< A reigon of our star
};

/**
 * An imaginary function that allocates a certain reigon of nowhere.
 *
 * @param size The size of the reigon.
 * @param type The located nowhere.
 */
cdoxy_data1 *cdoxy_allocate(int size, cdoxy_nowhere_in type);

///@}
