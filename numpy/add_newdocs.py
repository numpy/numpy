
from lib import add_newdoc

add_newdoc('numpy.core','dtype',
           [('fields', "Fields of the data-typedescr if any."),
            ('alignment', "Needed alignment for this data-type"),
            ('byteorder',
             "Little-endian (<), big-endian (>), native (=), or "\
             "not-applicable (|)"),
            ('char', "Letter typecode for this descriptor"),
            ('dtype', "Typeobject associated with this descriptor"),
            ('kind', "Character giving type-family of this descriptor"),
            ('itemsize', "Size of each item"),
            ('num', "Internally-used number for builtin base")
            ]
           )
