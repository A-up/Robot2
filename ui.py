import tkinter as tk
from tkinter import ttk
import subprocess

process = None

def toggle_script():
    global process

    if process is None or process.poll() is not None:
        # запуск
        process = subprocess.Popen([".venv/Scripts/python.exe", "cv_new.py"])
        status.config(text="Работает", fg="#00ff88")
        start_button.config(text="СТОП", bg="#e74c3c")
        progress.start(10)

    else:
        # остановка
        process.terminate()
        process = None
        status.config(text="Остановлено", fg="#aaaaaa")
        start_button.config(text="СТАРТ", bg="#2ecc71")
        progress.stop()


TK = tk.Tk()
TK.title("RM002 Control")
TK.geometry("500x300")
TK.configure(bg="#1e1e1e")

title = tk.Label(
    TK,
    text="Робот RM002\nКомпьютерное зрение\nг. Удомля • 2026",
    font=("Segoe UI", 18, "bold"),
    bg="#1e1e1e",
    fg="white",
    justify="center"
)
title.pack(fill="x", pady=15)

status = tk.Label(
    TK,
    text="Ожидание",
    font=("Segoe UI", 12),
    bg="#1e1e1e",
    fg="#aaaaaa"
)
status.pack(pady=5)

progress = ttk.Progressbar(
    TK,
    mode="indeterminate",
    length=300
)
progress.pack(pady=5)

start_button = tk.Button(
    TK,
    text="СТАРТ",
    font=("Segoe UI", 18, "bold"),
    bg="#2ecc71",
    fg="black",
    activebackground="#27ae60",
    relief="flat",
    command=toggle_script
)
start_button.pack(fill="x", padx=60, pady=20, ipady=15)

TK.mainloop()