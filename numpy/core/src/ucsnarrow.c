
/* the ucs2 buffer must be large enough to hold 2*ucs4length characters
   due to the use of surrogate pairs. 

   The return value is the number of ucs2 bytes used-up which
   is ucs4length + number of surrogate pairs found. 

   values above 0xffff are converted to surrogate pairs. 
 */
static int
PyUCS2Buffer_FromUCS4(Py_UNICODE *ucs2, PyArray_UCS4 *ucs4, int ucs4length)
{
        register int i;
        int surrpairs = 0;
        PyArray_UCS4 chr;
        for (i=0; i<ucs4length; i++) {
                chr = *ucs4++;
                if (chr > 0xffff) {
                        surrpairs++;
                        chr -= 0x10000L;
                        *ucs2++ = 0xD800 + (Py_UNICODE) (chr >> 10);
                        *ucs2++ = 0xDC00 + (Py_UNICODE) (chr & 0x03FF);
                }
                else {
                        *ucs2++ = (Py_UNICODE) (chr);
                }
        }
        return ucs4length + surrpairs;
}


/* This converts a UCS2 buffer (from a Python unicode object)

*/


static int
PyUCS2Buffer_AsUCS4(Py_UNICODE *ucs2, PyArray_UCS4 *ucs4, int ucs4length, int ucs2length)
{
}
