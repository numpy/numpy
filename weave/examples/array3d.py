""" A simple example to show how to access a 3D Numeric array.  One
example shows how to access the Numeric array using blitz type
converters and the other shows how it can be done without using blitz
by accessing the Numeric array data directly.

"""

import weave
from weave import converters
import Numeric

def create_array():
    """Creates a simple 3D Numeric array with unique values at each
    location in the matrix.

    """    
    rows, cols, depth = 2, 3, 4
    arr = Numeric.zeros((rows, cols, depth), 'i')
    count = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                arr[i,j,k] = count
                count += 1
    return arr


def pure_inline(arr):
    """Prints the given 3D array by accessing the raw Numeric data and
    without using blitz converters.

    Notice the following:
      1. '\\n' to escape generating a newline in the C++ code.
      2. rows, cols = Narr[0], Narr[1].
      3. Array access using arr[(i*cols + j)*depth + k].
      
    """
    
    code = """
    int rows = Narr[0];
    int cols = Narr[1];
    int depth = Narr[2];    
    for (int i=0; i < rows; i++)
    {
        for (int j=0; j < cols; j++)
        {
            printf("img[%3d][%3d]=", i, j);
            for (int k=0; k< depth; ++k)
            {
                printf(" %3d", arr[(i*cols + j)*depth + k]);
            }
            printf("\\n");
        }
    }
    """

    weave.inline(code, ['arr'])


def blitz_inline(arr):
    """Prints the given 3D array by using blitz converters which
    provides a Numeric-like syntax for accessing the Numeric data.

    Notice the following:
      1. '\\n' to escape generating a newline in the C++ code.
      2. rows, cols = Narr[0], Narr[1].
      3. Array access using arr(i, j, k).
      
    """
    
    code = """
    int rows = Narr[0];
    int cols = Narr[1];
    int depth = Narr[2];    
    for (int i=0; i < rows; i++)
    {
        for (int j=0; j < cols; j++)
        {
            printf("img[%3d][%3d]=", i, j);
            for (int k=0; k< depth; ++k)
            {
                printf(" %3d", arr(i, j, k));
            }
            printf("\\n");
        }
    }
    """

    weave.inline(code, ['arr'], type_converters=converters.blitz)


def main():
    arr = create_array()
    print "Numeric:"    
    print arr

    print "Pure Inline:"
    pure_inline(arr)
    
    print "Blitz Inline:"
    blitz_inline(arr)
    

if __name__ == '__main__':
    main()
