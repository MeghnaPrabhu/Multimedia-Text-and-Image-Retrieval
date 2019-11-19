import tkinter as Tkinter

from PIL import Image, ImageTk


class PictureWindow(Tkinter.Canvas):
    def __init__(self, image_list, *args, **kwargs):
        Tkinter.Canvas.__init__(self, *args, **kwargs)
        self.imagelist = image_list
        self.imagelist_p = []
        self.all_function_trigger()

    def tk_image(self, path, w, h):
        img = Image.open(path)
        img = img.resize((w, h))
        storeobj = ImageTk.PhotoImage(img)
        return storeobj

    def show_image(self, path):
        img = self.tk_image(path, self.winfo_screenwidth(), self.winfo_screenheight())
        self.delete(self.find_withtag("bacl"))
        self.allready = self.create_image(self.winfo_screenwidth() / 4, self.winfo_screenheight() / 4, image=img,
                                          anchor='center', tag="bacl")

        self.image = img
        print
        self.find_withtag("bacl")
        self.master.title("Image Viewer ({})".format(path))
        return

    def previous_image(self):
        try:
            pop = self.imagelist_p.pop()
            self.show_image(pop)
            self.imagelist.append(pop)
        except:
            pass
        return

    def next_image(self):
        try:
            pop = self.imagelist.pop()

            self.show_image(pop)
            self.imagelist_p.append(pop)
        except EOFError as e:
            pass
        except IndexError as e1:
            pass
        return

    def all_function_trigger(self):
        self.create_buttons()
        self.window_settings()
        return

    def window_settings(self):
        self['width'] = self.winfo_screenwidth()
        self['height'] = self.winfo_screenheight()
        return

    def create_buttons(self):
        Tkinter.Button(self, text=" Next ", command=self.next_image).place(x=(self.winfo_screenwidth() / 1.1),
                                                                        y=(self.winfo_screenheight() / 2))
        Tkinter.Button(self, text=" Prev ", command=self.previous_image).place(x=20, y=(self.winfo_screenheight() / 2))
        self['bg'] = "white"
        return