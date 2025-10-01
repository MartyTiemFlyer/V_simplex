import tkinter as tk
from tkinter import scrolledtext

class SimplexApp:
    def __init__(self, solver):
        self.solver = solver
        self.solver.solve()  # собираем шаги
        self.steps = solver.steps
        self.current = 0

        self.root = tk.Tk()
        self.root.title("Симплекс-метод")

        self.text_box = scrolledtext.ScrolledText(self.root, width=80, height=25)
        self.text_box.pack()

        frame = tk.Frame(self.root)
        frame.pack()

        self.btn_prev = tk.Button(frame, text="Назад", command=self.show_prev)
        self.btn_prev.grid(row=0, column=0)

        self.btn_next = tk.Button(frame, text="Далее", command=self.show_next)
        self.btn_next.grid(row=0, column=1)

        self.show_step()

        self.root.mainloop()

    def show_step(self):
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, self.steps[self.current])

    def show_prev(self):
        if self.current > 0:
            self.current -= 1
            self.show_step()

    def show_next(self):
        if self.current < len(self.steps) - 1:
            self.current += 1
            self.show_step()
