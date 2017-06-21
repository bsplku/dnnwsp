#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tkinter import *
from tkinter import ttk


# Save values into variables and close the widget
def get_value_ft():
    global mode
    global optimizer_algorithm
    global momtentum
    global nodes
    global n_epochs
    global batch_size
    global beginAnneal
    global decay_rate
    global lr_init
    global lr_min
    global beta_lrates
    global L2_reg
    global max_beta
    global tg_hspset
    
    mode=RadioButtonVar.get() 
    optimizer_algorithm=e2.get()
    momtentum=e3.get()
    nodes=[int(i) for i in e4.get().split(' ')]
    n_epochs=int(e5.get())
    batch_size=int(e6.get())
    beginAnneal=int(e7.get())
    decay_rate=float(e8.get())
    lr_init=float(e9.get())
    lr_min=float(e10.get())
    beta_lrates=float(e11.get())
    L2_reg=float(e12.get())
    max_beta=[float(i) for i in e13.get().split(' ')]
    tg_hspset=[float(i) for i in e14.get().split(' ')]
    
    
    master.destroy()    
    return
    

# Create a new Toplevel widget
master = Tk()
master.title("Customization GUI")

# Construct a string variable
RadioButtonVar = StringVar(None, 'layer') 
ComboBoxVar = StringVar()


#master.withdraw()
#master.update_idletasks()  # Update "requested size" from geometry manager
#
#x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
#y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
#    master.geometry('+%d+%d' % (x, y))



# Construct labels with parent master 
ttk.Label(master, text="Mode").grid(row=1)
ttk.Label(master, text="Optimizer algorithm").grid(row=3)
ttk.Label(master, text="Momtentum").grid(row=4)
ttk.Label(master, text="Nodes").grid(row=5)
ttk.Label(master, text="Total epoch").grid(row=6)
ttk.Label(master, text="Batch size").grid(row=7)
ttk.Label(master, text="BeginAnneal").grid(row=8)
ttk.Label(master, text="Decay rate").grid(row=9)
ttk.Label(master, text="Initial learning rate").grid(row=10)
ttk.Label(master, text="Minimum learning rate").grid(row=11)
ttk.Label(master, text="Beta learning rate").grid(row=12)
ttk.Label(master, text="L2 regularization parameter").grid(row=13)
ttk.Label(master, text="Max beta").grid(row=14)
ttk.Label(master, text="Target hsp").grid(row=15)

# Create buttons, combo box, and entries
e0=ttk.Radiobutton(master, text="Layer wise", value='layer', variable=RadioButtonVar)
e1=ttk.Radiobutton(master, text="Node wise", value='node', variable=RadioButtonVar)
e2 = ttk.Combobox(master, textvariable=ComboBoxVar)
e2['values'] = ('GradientDescent', 'Adagrad', 'Momentum','Adam','RMSProp') 
e3 = ttk.Entry(master)
e4 = ttk.Entry(master)
e5 = ttk.Entry(master)
e6 = ttk.Entry(master)
e7 = ttk.Entry(master)
e8 = ttk.Entry(master)
e9 = ttk.Entry(master)
e10 = ttk.Entry(master) 
e11 = ttk.Entry(master)
e12 = ttk.Entry(master)
e13 = ttk.Entry(master)
e14 = ttk.Entry(master)

########### Layer
# Insert the inital values at indices
e2.current(0)                       # optimizer algorithm
e3.insert(0, 0.01)                      # Momentum
e4.insert(0,[74484,100,100,100,4]) # nodes
e5.insert(0, 300)                  # total epoch
e6.insert(0, 40)                  # batch size
e7.insert(0, 50)                  # beginAnneal
e8.insert(0, 0.0005)                 # decay rate
e9.insert(0, 1e-3)                 # initial learning rate
e10.insert(0, 1e-4)                 # minimum learning rate
e11.insert(0, 1e-2)                # learning rate of beta
e12.insert(0, 1e-4)                # L2 parameter
e13.insert(0, [0.05, 0.95, 0.7])    # max beta
e14.insert(0, [0.7, 0.7, 0.5])     # target hsp               
          
          
          
########## Node          
## Insert the inital values at indices
#e2.current(0)                       # optimizer algorithm
#e3.insert(0, 0.5)                      # Momentum
#e4.insert(0,[74484,100,100,100,4]) # nodes
#e5.insert(0, 300)                  # total epoch
#e6.insert(0, 40)                  # batch size
#e7.insert(0, 50)                  # beginAnneal
#e8.insert(0, 0.0005)                 # decay rate
#e9.insert(0, 1e-3)                 # initial learning rate
#e10.insert(0, 1e-4)                 # minimum learning rate
#e11.insert(0, 1e-2)                # learning rate of beta
#e12.insert(0, 1e-4)                # L2 parameter
#e13.insert(0, [0.05, 0.8, 0.8])    # max beta
#e14.insert(0, [0.7, 0.5, 0.5])     # target hsp               
#          
          
          
          
          
          
          
# Position a widget in the parent widget in a grid. 
e0.grid(row=1, column=1)
e1.grid(row=2, column=1)
e2.grid(row=3, column=1)
e3.grid(row=4, column=1)
e4.grid(row=5, column=1)
e5.grid(row=6, column=1)
e6.grid(row=7, column=1)
e7.grid(row=8, column=1)
e8.grid(row=9, column=1)
e9.grid(row=10, column=1)
e10.grid(row=11, column=1)
e11.grid(row=12, column=1)
e12.grid(row=13, column=1)
e13.grid(row=14, column=1)
e14.grid(row=14, column=1)


# Ttk Button widget evaluates a command when pressed
ttk.Button(master, text='Done', command=get_value_ft).grid(row=15, column=1, pady=10)

# Call the mainloop
master.mainloop()
