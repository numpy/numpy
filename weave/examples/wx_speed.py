""" Implements a fast replacement for calling DrawLines with an array as an
    argument.  It uses weave, so you'll need that installed.

    Copyright:   Space Telescope Science Institute
    License:     BSD Style
    Designed by: Enthought, Inc.
    Author:      Eric Jones eric@enthought.com

    I wrote this because I was seeing very bad performance for DrawLines when
    called with a large number of points -- 5000-30000. Now, I have found the
    performance is sometimes OK, and sometimes very poor.  Drawing to a
    MemoryDC seems to be worse than drawing to the screen.  My first cut of the
    routine just called PolyLine directly, but I got lousy performance for this
    also.  After noticing the slowdown as the array length grew was much worse
    than linear, I tried the following "chunking" algorithm.  It is much more
    efficient (sometimes by 2 orders of magnitude, but usually only a factor
    of 3).  There is a slight drawback in that it will draw end caps for each
    chunk of the array which is not strictly correct.  I don't imagine this is
    a major issue, but remains an open issue.

"""
import weave
from RandomArray import *
from Numeric import *
from wxPython.wx import *

"""
const int n_pts = _Nline[0];
const int bunch_size = 100;
const int bunches = n_pts / bunch_size;
const int left_over = n_pts % bunch_size;

for (int i = 0; i < bunches; i++)
{
    Polyline(hdc,(POINT*)p_data,bunch_size);
    p_data += bunch_size*2; //*2 for two longs per point
}
Polyline(hdc,(POINT*)p_data,left_over);
"""

def polyline(dc,line,xoffset=0,yoffset=0):
    #------------------------------------------------------------------------
    # Make sure the array is the correct size/shape 
    #------------------------------------------------------------------------
    shp = line.shape
    assert(len(shp)==2 and shp[1] == 2)

    #------------------------------------------------------------------------
    # Offset data if necessary
    #------------------------------------------------------------------------
    if xoffset or yoffset:
        line = line + array((xoffset,yoffset),line.typecode())
    
    #------------------------------------------------------------------------
    # Define the win32 version of the function
    #------------------------------------------------------------------------        
    if sys.platform == 'win32':
        # win32 requires int type for lines.
        if (line.typecode() != Int or not line.iscontiguous()):
            line = line.astype(Int)   
        code = """
               HDC hdc = (HDC) dc->GetHDC();                    
               Polyline(hdc,(POINT*)line,Nline[0]);
               """
    else:
        if (line.typecode() != UInt16 or 
            not line.iscontiguous()):
            line = line.astype(UInt16)   
        code = """
               GdkWindow* win = dc->m_window;                    
               GdkGC* pen = dc->m_penGC;
               gdk_draw_lines(win,pen,(GdkPoint*)line,Nline[0]);         
               """
    weave.inline(code,['dc','line'])

    
    #------------------------------------------------------------------------
    # Find the maximum and minimum points in the drawing list and add
    # them to the bounding box.    
    #------------------------------------------------------------------------
    max_pt = maximum.reduce(line,0)
    min_pt = minimum.reduce(line,0)
    dc.CalcBoundingBox(max_pt[0],max_pt[1])
    dc.CalcBoundingBox(min_pt[0],min_pt[1])    

#-----------------------------------------------------------------------------
# Define a new version of DrawLines that calls the optimized
# version for Numeric arrays when appropriate.
#-----------------------------------------------------------------------------
def NewDrawLines(dc,line):
    """
    """
    if (type(line) is ArrayType):
        polyline(dc,line)
    else:
        dc.DrawLines(line)            

#-----------------------------------------------------------------------------
# And attach our new method to the wxPaintDC class
# !! We have disabled it and called polyline directly in this example
# !! to get timing comparison between the old and new way.
#-----------------------------------------------------------------------------
#wxPaintDC.DrawLines = NewDrawLines
        
if __name__ == '__main__':
    from wxPython.wx import *
    import time

    class Canvas(wxWindow):
        def __init__(self, parent, id = -1, size = wxDefaultSize):
            wxWindow.__init__(self, parent, id, wxPoint(0, 0), size,
                              wxSUNKEN_BORDER | wxWANTS_CHARS)
            self.calc_points()
            EVT_PAINT(self, self.OnPaint)
            EVT_SIZE(self, self.OnSize)

        def calc_points(self):
            w,h = self.GetSizeTuple()            
            #x = randint(0+50, w-50, self.point_count)
            #y = randint(0+50, h-50, len(x))
            x = arange(0,w,typecode=Int32)
            y = h/2.*sin(x*2*pi/w)+h/2.
            y = y.astype(Int32)
            self.points = concatenate((x[:,NewAxis],y[:,NewAxis]),-1)

        def OnSize(self,event):
            self.calc_points()
            self.Refresh()

        def OnPaint(self,event):
            w,h = self.GetSizeTuple()            
            print len(self.points)
            dc = wxPaintDC(self)
            dc.BeginDrawing()

            # This first call is slow because your either compiling (very slow)
            # or loading a DLL (kinda slow)
            # Resize the window to get a more realistic timing.
            pt_copy = self.points.copy()
            t1 = time.clock()
            offset = array((1,0))
            mod = array((w,0))
            x = pt_copy[:,0];
            ang = 2*pi/w;
            
            size = 1
            red_pen = wxPen('red',size)            
            white_pen = wxPen('white',size)
            blue_pen = wxPen('blue',size)
            pens = iter([red_pen,white_pen,blue_pen])
            phase = 10
            for i in range(1500):
                if phase > 2*pi:
                    phase = 0                
                    try:
                        pen = pens.next()
                    except:
                        pens = iter([red_pen,white_pen,blue_pen])
                        pen = pens.next()
                    dc.SetPen(pen)
                polyline(dc,pt_copy)            
                next_y = (h/2.*sin(x*ang-phase)+h/2.).astype(Int32)            
                pt_copy[:,1] = next_y
                phase += ang
            t2 = time.clock()
            print 'Weave Polyline:', t2-t1

            t1 = time.clock()
            pt_copy = self.points.copy()
            pens = iter([red_pen,white_pen,blue_pen])
            phase = 10
            for i in range(1500):
                if phase > 2*pi:
                    phase = 0                
                    try:
                        pen = pens.next()
                    except:
                        pens = iter([red_pen,white_pen,blue_pen])
                        pen = pens.next()
                    dc.SetPen(pen)
                dc.DrawLines(pt_copy)
                next_y = (h/2.*sin(x*ang-phase)+h/2.).astype(Int32)            
                pt_copy[:,1] = next_y
                phase += ang
            t2 = time.clock()
            dc.SetPen(red_pen)
            print 'wxPython DrawLines:', t2-t1

            dc.EndDrawing()

    class CanvasWindow(wxFrame):
        def __init__(self, id=-1, title='Canvas',size=(500,500)):
            parent = NULL
            wxFrame.__init__(self, parent,id,title, size=size)
            self.canvas = Canvas(self)
            self.Show(1)

    class MyApp(wxApp):
        def OnInit(self):
            frame = CanvasWindow(title="Speed Examples",size=(500,500))
            frame.Show(true)
            return true

    app = MyApp(0)
    app.MainLoop()
    