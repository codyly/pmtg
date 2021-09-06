"""PMTG Synthetic Control Problem: Draw Synthetic Controll Trajectory

The trajectory should be drawed in the correct order
*** 1. start from a point closed to the origin
*** 2. go through No. 1,4,2,3 quadrants in turn
*** 3. return to a point closed to the start point

Author: Ren Liu
"""

from tkinter import Tk, Button, Scale, Canvas, HORIZONTAL, RAISED, SUNKEN, ROUND, TRUE

import numpy as np


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = "black"

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text="pen", command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg="white", width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        # initialize an empty list for logging trajectory points
        self.log = []

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind("<B1-Motion>", self.paint)
        self.c.bind("<ButtonRelease-1>", self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(
                self.old_x,
                self.old_y,
                event.x,
                event.y,
                width=self.line_width,
                fill=paint_color,
                capstyle=ROUND,
                smooth=TRUE,
                splinesteps=36,
            )
        self.old_x = event.x
        self.old_y = event.y

        # append trajectory points into log array
        self.log.append([event.x, event.y])

    def reset(self, event):

        # update the local trajectory files
        np.save("trajectory.npy", self.log)

        self.old_x, self.old_y = None, None


if __name__ == "__main__":
    Paint()
