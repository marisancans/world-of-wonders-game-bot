import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')

from gi.repository import Gdk
from gi.repository import GdkPixbuf
import numpy as np
from Xlib.display import Display

class ScreenShot():

    def __init__(self) -> None:
        #define the window name
        window_name = 'Mi A2 Lite'

        window = Gdk.get_default_root_window()
        screen = window.get_screen()
        stack = screen.get_window_stack()
        myselectwindow = self.locate_window(stack, window_name)
        self.img_pixbuf = Gdk.pixbuf_get_from_window(myselectwindow, *myselectwindow.get_geometry()) 

    #define xid of your select 'window'
    def locate_window(self, stack, window):
        disp = Display()
        NET_WM_NAME = disp.intern_atom('_NET_WM_NAME')
        WM_NAME = disp.intern_atom('WM_NAME') 
        name= []
        for i, w in enumerate(stack):
            win_id =w.get_xid()
            window_obj = disp.create_resource_object('window', win_id)
            for atom in (NET_WM_NAME, WM_NAME):
                window_name=window_obj.get_full_property(atom, 0)
                name.append(window_name.value)
        for l in range(len(stack)):
            n = name[2*l].decode("utf-8")
            if(n==window):
                return stack[l]

    def pixbuf_to_array(self, p):
        w,h,c,r=(p.get_width(), p.get_height(), p.get_n_channels(), p.get_rowstride())
        assert p.get_colorspace() == GdkPixbuf.Colorspace.RGB
        assert p.get_bits_per_sample() == 8
        if  p.get_has_alpha():
            assert c == 4
        else:
            assert c == 3
        assert r >= w * c
        a=np.frombuffer(p.get_pixels(),dtype=np.uint8)
        if a.shape[0] == w*c*h:
            return a.reshape( (h, w, c) )
        else:
            b=np.zeros((h,w*c),'uint8')
            for j in range(h):
                b[j,:]=a[r*j:r*j+w*c]
            return b.reshape( (h, w, c) )

    def get_screen(self):
        beauty_print = self.pixbuf_to_array(self.img_pixbuf)
        return beauty_print