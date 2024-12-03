import tkinter as tk
from tkinter import filedialog, messagebox

def open_dialog():
    def selectFileOrDirectory():
        choice = selection_var.get()
        if choice == 'File':
            path = filedialog.askopenfilename()
            if path:
                messagebox.showinfo("Selected", f"File selected: {path}")
        elif choice == 'Directory':
            path = filedialog.askdirectory()
            if path:
                messagebox.showinfo("Selected", f"Directory selected: {path}")

    # Create the root window and hide it
    window = tk.Tk()
    window.withdraw()
    
    # Create a top-level dialog
    button_imgOrDir = tk.Toplevel(window)
    button_imgOrDir.title("Select an Input Image or Directory of Images")

    # Variable to store the user choice
    selection_var = tk.StringVar(value='File')

    # Drop-down menu for selecting file or directory
    choice_menu = tk.OptionMenu(button_imgOrDir, selection_var, 'File', 'Directory')
    choice_menu.pack(pady=10)

    # Button to confirm the selection
    confirm_button = tk.Button(button_imgOrDir, text="Open", command=selectFileOrDirectory)
    confirm_button.pack(pady=10)

    # Run the Tkinter event loop
    window.mainloop()

open_dialog()