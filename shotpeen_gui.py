import tkinter as tk
from tkinter import filedialog, ttk, messagebox


def check_install(package_id: str):
    try:
        __import__(package_id)
    except ModuleNotFoundError:
        print(f"{package_id} not installed")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_id])

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Model GUI")
        self.root.geometry("800x600")
        self.main_menu()

    def main_menu(self):
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Main Menu Layout
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True)

        tk.Label(main_frame, text="Main Menu", font=("Arial", 24)).grid(row=0, column=0, columnspan=2, pady=20)

        tk.Button(main_frame, text="Train Model", command=self.train_model_dialog, width=20, height=2).grid(row=1, column=0, padx=20, pady=20)
        tk.Button(main_frame, text="Load Model", command=self.load_model_dialog, width=20, height=2).grid(row=1, column=1, padx=20, pady=20)

    def train_model_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Train Model")
        dialog.geometry("700x500")

        # Training Layout
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(expand=True, fill=tk.BOTH)

        # File Selection
        tk.Label(frame, text="Training and Testing Data", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=10)
        data_file_var = tk.StringVar()
        tk.Entry(frame, textvariable=data_file_var, width=50).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(frame, text="Browse", command=lambda: self.browse_file(data_file_var)).grid(row=0, column=2, pady=5)

        # Log and Progress Bar
        tk.Label(frame, text="Training Log", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=10)
        log = tk.Text(frame, height=10, width=60, state='disabled')
        log.grid(row=2, column=0, columnspan=3, pady=10)

        progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=500, mode='determinate')
        progress.grid(row=3, column=0, columnspan=3, pady=10)

        # Buttons
        tk.Button(frame, text="Train", command=lambda: self.start_training(log, progress), width=15).grid(row=4, column=0, pady=20)
        tk.Button(frame, text="Back to Main Menu", command=dialog.destroy, width=15).grid(row=4, column=2, pady=20)

    def load_model_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Model")
        dialog.geometry("700x500")

        # Load Model Layout
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(expand=True)

        # Configure grid to center contents
        for i in range(5):  # Assuming there are up to 5 rows
            frame.grid_rowconfigure(i, weight=1)
        for i in range(3):  # Assuming 3 columns: Label, Entry, and Button
            frame.grid_columnconfigure(i, weight=1)

        # Model File Selection
        tk.Label(frame, text="Model File", font=("Arial", 12)).grid(row=0, column=0, sticky="e", pady=10)
        model_file_var = tk.StringVar()
        tk.Entry(frame, textvariable=model_file_var, width=50).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(frame, text="Browse", command=lambda: self.browse_file(model_file_var)).grid(row=0, column=2, pady=5)

        # STEP File Selection
        tk.Label(frame, text="STEP File", font=("Arial", 12)).grid(row=1, column=0, sticky="e", pady=10)
        step_file_var = tk.StringVar()
        tk.Entry(frame, textvariable=step_file_var, width=50).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(frame, text="Browse", command=lambda: self.browse_file(step_file_var)).grid(row=1, column=2, pady=5)

        # Output Path Selection
        tk.Label(frame, text="Output Path", font=("Arial", 12)).grid(row=2, column=0, sticky="e", pady=10)
        output_path_var = tk.StringVar()
        tk.Entry(frame, textvariable=output_path_var, width=50).grid(row=2, column=1, padx=10, pady=5)
        tk.Button(frame, text="Browse", command=lambda: self.browse_directory(output_path_var)).grid(row=2, column=2, pady=5)

        # Buttons at the bottom
        tk.Button(frame, text="Preview STEP File", command=lambda: self.preview_file(step_file_var.get()), width=20).grid(row=3, column=1, pady=20, sticky="n")
        tk.Button(frame, text="Back to Main Menu", command=dialog.destroy, width=15).grid(row=4, column=1, pady=20, sticky="n")

    def browse_file(self, variable):
        filepath = filedialog.askopenfilename()
        if filepath:
            variable.set(filepath)

    def browse_directory(self, variable):
        dirpath = filedialog.askdirectory()
        if dirpath:
            variable.set(dirpath)

    def preview_file(self, file_path):
        if file_path:
            messagebox.showinfo("Preview", f"Previewing: {file_path}")
        else:
            messagebox.showerror("Error", "No file selected!")

    def start_training(self, log_widget, progress_bar):
        # Simulate training process
        log_widget.config(state='normal')
        log_widget.insert(tk.END, "Training started...\n")
        log_widget.see(tk.END)
        log_widget.config(state='disabled')
        progress_bar.start(10)  # Simulate progress
        self.root.after(3000, lambda: self.finish_training(log_widget, progress_bar))

    def finish_training(self, log_widget, progress_bar):
        progress_bar.stop()
        log_widget.config(state='normal')
        log_widget.insert(tk.END, "Training completed!\n")
        log_widget.see(tk.END)
        log_widget.config(state='disabled')


# Check for missing modules
dependencies = [
    "requests>=2.25.1",
    "numpy>=1.20.0",
    "matplotlib>=3.4.0",
    "pandas>=1.3.0",
    "torch>=1.9.0",
    "tkinter"
]

for library in dependencies:
    check_install(library)

# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
