import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from PIL import Image
from PIL import ImageTk


class UserInterface(tk.Tk):

    def __init__(self, org_img_arr: np.array, resize_shape=(320, 320), prediction='', exp_mask_arr=None):
        """
        constructor

        :param org_img_arr: array of original image
        :param prediction: prediction, class
        :param resize_shape: size for resizing
        :param exp_mask_arr: array of explanation, if None random array is generated
        """

        super().__init__()
        self.resize_shape = resize_shape

        ###
        # user input
        ###

        # prepare original image for tkinter
        org_image = Image.fromarray(org_img_arr)
        org_image = org_image.resize(self.resize_shape, Image.ANTIALIAS)
        org_image = ImageTk.PhotoImage(org_image)

        # create default explanation, if explanation is empty
        if exp_mask_arr is None:
            exp_mask_arr = np.zeros((org_img_arr.shape[0], org_img_arr.shape[1]))
            exp_mask_arr[25:50, 25:50] = 1
            exp_mask_arr = exp_mask_arr.astype(bool)

        # unite original image with explanation
        # convert to RGB to draw light green explanations
        exp_img_arr = Image.fromarray(org_img_arr)
        exp_img_arr = exp_img_arr.convert('RGB')
        exp_img_arr = np.asarray(exp_img_arr)
        exp_img_arr = np.copy(exp_img_arr)  # copy image array, since it set from PIL to non-writeable otherwise
        exp_img_arr[exp_mask_arr, 1] = 255

        # prepare explanation mask for tkinter
        exp_image = Image.fromarray(exp_img_arr)
        exp_image = exp_image.resize(self.resize_shape, Image.ANTIALIAS)
        exp_image = ImageTk.PhotoImage(exp_image)

        # initialize user input
        self.org_img_arr = org_img_arr
        self.org_shape = org_img_arr.shape
        self.org_image = org_image
        self.prediction = prediction
        self.exp_mask_arr = exp_mask_arr
        self.exp_img_arr = exp_img_arr
        self.exp_image = exp_image

        ###
        # interface
        ###

        # basic definition
        self.result = None
        self.correction_mask = None
        self.drawing_mode = False
        self.drawing_started = False
        self.image_mode = "org"
        self.string_var = tk.StringVar()
        self.point_size = 15
        self.drawn_elements = []
        self.drawn_points = []
        self.drawn_mask = np.zeros(self.resize_shape)

        # window
        self.screen_w = self.winfo_screenwidth()
        self.geometry(f"420x450+{self.screen_w}+0")

        # elements
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        self.button1 = None
        self.button2 = None
        self.button3 = None
        self.button4 = None
        self.button5 = None
        self.button6 = None
        self.button7 = None
        self.button8 = None
        self.button9 = None
        self.label1 = None
        self.label2 = None
        self.canvas = None
        self.canvas_img = None

        self.create_elements()

    def create_elements(self):
        """
        creates all tk objects of the interface
        """

        self.frame1 = tk.Frame(self)
        self.frame1.grid(row=0, column=0)

        self.frame2 = tk.Frame(self)
        self.frame2.grid(row=0, column=1)

        self.frame3 = tk.Frame(self)

        self.label1 = tk.Label(self.frame1, text=f"Prediction: {self.prediction}")
        self.label1.pack()

        self.button1 = ttk.Button(self.frame1, text="Explanation", command=self.button1_addition)
        self.button1.pack()

        self.canvas = tk.Canvas(self.frame1, height=319, width=319)
        self.canvas_img = self.canvas.create_image(0, 0, image=self.org_image, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing())
        self.canvas.pack()

        self.button2 = ttk.Button(self.frame2, text="True(RR)", command=self.button2_addition)
        self.button2.pack()

        self.button3 = ttk.Button(self.frame2, text="True(WR)", command=self.button3_addition)
        self.button3.pack()

        self.button4 = ttk.Button(self.frame2, text="False(W)", command=self.button4_addition)
        self.button4.pack()

        self.button5 = ttk.Button(self.frame3, text="Confirm", command=self.button5_addition)
        self.button5.pack(side=tk.LEFT)

        self.button6 = ttk.Button(self.frame3, text="Preview", command=self.button6_addition)
        self.button6.pack(side=tk.LEFT)

        self.button7 = ttk.Button(self.frame3, text="Reset", command=self.button7_addition)
        self.button7.pack(side=tk.LEFT)

        self.button8 = ttk.Button(self.frame3, text="Up", command=self.button8_addition)
        self.button8.pack()

        self.string_var.set(f"pencil width = {self.point_size}")
        self.label2 = tk.Label(self.frame3, textvariable=self.string_var)
        self.label2.pack()

        self.button9 = ttk.Button(self.frame3, text="Down", command=self.button9_addition)
        self.button9.pack()

    def start_drawing(self, event):
        """
        starts drawing mode, cursor on image, draws rectangles
        
        :param event: tkinter event
        """

        if self.drawing_mode:
            self.drawing_started = True

            # draw rectangle on canvas
            rect = self.canvas.create_rectangle(
                event.x - self.point_size, event.y - self.point_size,
                event.x + self.point_size + 1, event.y + self.point_size + 1,
                fill='green', width=0
            )

            self.drawn_elements.append(rect)
            self.drawn_points.append((event.x, event.y))

    def draw(self, event):
        """
        continues drawing mode, cursor over image, draws rectangles
        
        :param event: tkinter event
        """

        if self.drawing_mode and self.drawing_started:

            # draw rectangle on canvas
            rect = self.canvas.create_rectangle(
                event.x - self.point_size, event.y - self.point_size,
                event.x + self.point_size + 1, event.y + self.point_size + 1,
                fill='green', width=0
            )

            self.drawn_elements.append(rect)
            self.drawn_points.append((event.x, event.y))

    def end_drawing(self):
        """
        ends drawing mode
        """
        
        if self.drawing_mode and self.drawing_started:
            self.drawing_started = False

    def button1_addition(self):
        """
        shows explanation button
        
        shows explanation if image is visible
        deletes explanation is button is clicked again
        """

        if self.image_mode == "org":
            self.image_mode = "exp"
            self.canvas.delete(self.canvas_img)
            self.canvas_img = self.canvas.create_image(0, 0, image=self.exp_image, anchor="nw")

        elif self.image_mode == "exp":
            self.image_mode = "org"
            self.canvas.delete(self.canvas_img)
            self.canvas_img = self.canvas.create_image(0, 0, image=self.org_image, anchor="nw")

        # reset drawn elements and drawn points list
        self.drawn_elements, self.drawn_points = [], []

    def button2_addition(self):
        """
        right for the right reason button
        
        stores result and closes app
        """
        
        self.result = "RR"
        self.correction_mask = self.exp_mask_arr
        self.destroy()

    def button3_addition(self):
        """
        right for the wrong reason button
        
        stores result and opens annotation mode
        """
        
        self.result = "RW"
        self.frame2.grid_forget()
        self.frame3.grid(row=1, column=0)
        self.drawing_mode = True
        self.canvas.configure(cursor="pencil")

    def button4_addition(self):
        """
        wrong button
        
        stores result and opens annotation mode
        """
        
        self.result = "W"
        self.frame2.grid_forget()
        self.frame3.grid(row=1, column=0)
        self.drawing_mode = True
        self.canvas.configure(cursor="pencil")

    def button5_addition(self):
        """
        confirmation button
        
        saves correction mask and closes app
        """
        drawn_mask = self.calc_drawn_mask()
        drawn_mask = Image.fromarray(drawn_mask)

        drawn_mask = drawn_mask.resize((self.org_shape[0], self.org_shape[1]))
        drawn_mask = np.asarray(drawn_mask)

        self.correction_mask = drawn_mask.astype(bool)
        
        self.destroy()

    def button6_addition(self):
        """
        preview button
        
        shows annotated image
        """
        
        drawn_mask = self.calc_drawn_mask()

        image = Image.fromarray(self.org_img_arr)
        image = image.convert('RGB')
        image = image.resize(self.resize_shape)
        image = np.asarray(image)

        image[drawn_mask, 1] = 255

        image = Image.fromarray(image)
        image = image.convert('RGB')

        image.show()

    def button7_addition(self):
        """
        reset button
        
        deletes drawn explanations
        """
        
        for ele in self.drawn_elements:
            self.canvas.delete(ele)
        self.drawn_elements, self.drawn_points = [], []

    def button8_addition(self):
        """
        up button
        
        increases pencil size
        """
        
        if self.point_size < 100:
            self.point_size += 1
            self.string_var.set(f"pencil width = {self.point_size}")

    def button9_addition(self):
        """
        down button
        
        decreases pencil size
        """
        if self.point_size > 1:
            self.point_size -= 1
            self.string_var.set(f"pencil width = {self.point_size}")

    def calc_drawn_mask(self):
        """
        method that unifies drawn elements in single explanation mask
        """

        drawn_mask = np.zeros(self.resize_shape)

        # iterate though all drawn points
        for (y, x) in self.drawn_points:

            # get total selected points by point and pencil width
            rect = [
                x - self.point_size,
                x + self.point_size + 1,
                y - self.point_size,
                y + self.point_size + 1
            ]

            # limit mask to original image size
            for i in range(len(rect)):
                if rect[i] > 320: 
                    rect[i] = 320
                elif rect[i] < 0: 
                    rect[i] = 0

            # new explanation points in mask are set to 1
            drawn_mask[rect[0]:rect[1], rect[2]:rect[3]] = 1

        return drawn_mask.astype(bool)

    def get_results(self):
        """
        method that returns the results self.result and self.correction_mask
        """

        return self.result, self.correction_mask


if __name__ == "__main__":

    img_path = 'database/interface_example/ChestCT_example.jpeg'
    org_img = Image.open(img_path)

    thresh = 127
    fn = lambda x: x if x > thresh else 0
    image_bin = org_img.convert('L').point(fn)
    image_bin_arr = np.asarray(image_bin)

    ui = UserInterface(org_img_arr=image_bin_arr, prediction='ChestCT')
    ui.button2_addition()
    result, corr_mask = ui.get_results()

    if result == 'RR':
        print(0)
    else:
        print(-1)
