
import tkinter as tk

# Create main window
root = tk.Tk()
root.title("Calculator")
root.geometry("350x350")

# Entry box
entry = tk.Entry(root, width=25, borderwidth=5, font=("Arial", 16))
entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# Function to insert numbers
def click(num):
    current = entry.get()
    entry.delete(0, tk.END)
    entry.insert(0, str(current) + str(num))

# Function to clear
def clear():
    entry.delete(0, tk.END)

# Function to calculate
def equal():
    try:
        result = eval(entry.get())
        entry.delete(0, tk.END)
        entry.insert(0, result)
    except:
        entry.delete(0, tk.END)
        entry.insert(0, "Error")

# Button layout
buttons = [
    ('7',1,0), ('8',1,1), ('9',1,2), ('/',1,3),
    ('4',2,0), ('5',2,1), ('6',2,2), ('*',2,3),
    ('1',3,0), ('2',3,1), ('3',3,2), ('-',3,3),
    ('0',4,0), ('C',4,1), ('=',4,2), ('+',4,3)
]

for (text,row,col) in buttons:
    if text == "C":
        tk.Button(root, text=text, width=5, height=2,
                  command=clear).grid(row=row,column=col)
    elif text == "=":
        tk.Button(root, text=text, width=5, height=2,
                  command=equal).grid(row=row,column=col)
    else:
        tk.Button(root, text=text, width=5, height=2,
                  command=lambda t=text: click(t)).grid(row=row,column=col)

root.mainloop()