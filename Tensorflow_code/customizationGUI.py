#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tkinter import *
from tkinter import ttk



def get_value_ft():
    global mode
    global optimizer_algorithm
    global nodes
    global total_epoch
    global batch_size
    global beginAnneal
    global decay_rate
    global lr_init
    global min_lr
    global lr_beta
    global L2_param
    global max_beta
    global tg_hsp
    
    mode=RadioButtonVar.get() 
    optimizer_algorithm=e2.get()
    nodes=[int(i) for i in e3.get().split(' ')]
    total_epoch=int(e4.get())
    batch_size=int(e5.get())
    beginAnneal=int(e6.get())
    decay_rate=float(e7.get())
    lr_init=float(e8.get())
    min_lr=float(e9.get())
    lr_beta=float(e10.get())
    L2_param=float(e11.get())
    max_beta=[float(i) for i in e12.get().split(' ')]
    tg_hsp=[float(i) for i in e13.get().split(' ')]
    
    
    master.destroy()    
    return
    
def close_window_ft():
    master.destroy()
    return


master = Tk()
master.title("Customization GUI")

RadioButtonVar = StringVar(None, 'layer') 
ComboBoxVar = StringVar()


#master.withdraw()
#master.update_idletasks()  # Update "requested size" from geometry manager
#
#x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
#y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
#    master.geometry('+%d+%d' % (x, y))




ttk.Label(master, text="Mode").grid(row=1)
ttk.Label(master, text="optimizer algorithm").grid(row=3)
ttk.Label(master, text="nodes").grid(row=4)
ttk.Label(master, text="total epoch").grid(row=5)
ttk.Label(master, text="batch size").grid(row=6)
ttk.Label(master, text="beginAnneal").grid(row=7)
ttk.Label(master, text="decay rate").grid(row=8)
ttk.Label(master, text="initial learning rate").grid(row=9)
ttk.Label(master, text="minimum learning rate").grid(row=10)
ttk.Label(master, text="learning rate of beta").grid(row=11)
ttk.Label(master, text="L2 parameter").grid(row=12)
ttk.Label(master, text="max beta").grid(row=13)
ttk.Label(master, text="target hsp").grid(row=14)


e0=ttk.Radiobutton(master, text="Layer wise", value='layer', variable=RadioButtonVar)
e1=ttk.Radiobutton(master, text="Node wise", value='node', variable=RadioButtonVar)
e2 = ttk.Combobox(master, textvariable=ComboBoxVar)
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




e2['values'] = ('GradientDescent', 'Adagrad', 'Momentum','Adam','RMSProp')
e2.current(0)
e3.insert(10,[74484,100,100,100,4])
e4.insert(10, 300)          
e5.insert(10, 100)          
e6.insert(10, 90)          
e7.insert(10, 1e-4)          
e8.insert(10, 1e-3)          
e9.insert(10, 1e-4)          
e10.insert(10, 0.02)          
e11.insert(10, 1e-5)
e12.insert(10, [0.05, 0.8, 0.8])  
e13.insert(10, [0.7, 0.65, 0.65])           

e0.grid(row = 1, column = 1)
e1.grid(row = 2, column = 1)
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


ttk.Button(master, text='Done', command=get_value_ft).grid(row=15, column=0, padx=50, pady=10)
ttk.Button(master, text='Quit', command=close_window_ft).grid(row=15, column=1, padx=0, pady=10)

master.mainloop()


#mode, optimizer_algorithm, nodes, total_epoch, batch_size, beginAnneal, decay_rate, lr_init, min_lr,lr_beta, L2_param, max_beta, tg_hsp
