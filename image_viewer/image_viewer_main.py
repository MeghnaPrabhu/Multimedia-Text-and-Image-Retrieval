from image_viewer.picture_viewer import PictureWindow
import tkinter as Tkinter


class ImageViewerMain:

    def start_image_viewer(self, image_list):
        root = Tkinter.Tk(className=" Image Viewer")
        # Creating Canvas Widget
        PictureWindow(image_list, root).pack(expand="yes", fill="both")
        # Not Resizable
        root.resizable(width=0, height=0)
        # Window Mainloop
        root.mainloop()
        return