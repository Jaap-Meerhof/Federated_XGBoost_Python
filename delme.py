import tkinter as tk
from tkinter import ttk
import time

def start_progress():
    num_waypoints = 3
    for i_main in range(num_waypoints+1):
        progress_var.set(i_main)
        progress_label.config(text=f"Main Progress: {i_main} / {num_waypoints}")
        progress_bar.update()
        for i_sec in range(101):
            progress_var_sec.set(i_sec)
            progress_label_sec.config(text=f"Doing first thing Progress: {i_sec}%")
            progress_bar_sec.update()
            time.sleep(0.1)

        

app = tk.Tk()
app.title("Progress GUI")

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(app, variable=progress_var, maximum=3)
progress_bar.pack(pady=20)

progress_label = tk.Label(app, text="Main Progress: 0%")
progress_label.pack()

progress_var_sec = tk.DoubleVar()
progress_bar_sec = ttk.Progressbar(app, variable=progress_var_sec, maximum=100)
progress_bar_sec.pack(pady=20)

progress_label_sec = tk.Label(app, text="Sec Progress: 0%")
progress_label_sec.pack()


# Start button
start_button = tk.Button(app, text="Start Progress", command=start_progress)
start_button.pack()

app.mainloop()
