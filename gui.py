
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import cv2

from video_procesor import read_video
from database_manager import (
    add_person,
    add_embedding,
    get_known_people
)


class FaceRecognitionGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("900x650")

        # --------------------------------
        # Color scheme
        # --------------------------------

        self.bg_color = "#1e1e1e"
        self.panel_color = "#252526"
        self.card_color = "#2d2d30"
        self.text_color = "#f1f1f1"
        self.secondary_text = "#b5b5b5"
        self.accent_color = "#3b82f6"
        self.button_color = "#2f6fed"

        self.root.configure(
            bg=self.bg_color
        )

        # --------------------------------
        # ttk styling
        # --------------------------------

        style = ttk.Style()

        style.theme_use("clam")

        style.configure(
            "TNotebook",
            background=self.bg_color,
            borderwidth=0
        )

        style.configure(
            "TNotebook.Tab",
            background=self.panel_color,
            foreground=self.secondary_text,
            padding=(15, 8)
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", self.accent_color)
            ],
            foreground=[
                ("selected", "#ffffff")
            ]
        )

        style.configure(
            "TFrame",
            background=self.bg_color
        )

        # --------------------------------
        # Image references
        # --------------------------------

        self.photo_refs = []

        # --------------------------------
        # Notebook
        # --------------------------------

        self.notebook = ttk.Notebook(
            self.root
        )

        self.notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )

        # --------------------------------
        # Tabs
        # --------------------------------

        self.video_tab = tk.Frame(
            self.notebook,
            bg=self.bg_color
        )

        self.add_person_tab = tk.Frame(
            self.notebook,
            bg=self.bg_color
        )

        self.known_people_tab = tk.Frame(
            self.notebook,
            bg=self.bg_color
        )

        self.notebook.add(
            self.video_tab,
            text="Video Recognition"
        )

        self.notebook.add(
            self.add_person_tab,
            text="Add Person"
        )

        self.notebook.add(
            self.known_people_tab,
            text="Known People"
        )

        # --------------------------------
        # Create tabs
        # --------------------------------

        self.create_video_tab()
        self.create_add_person_tab()
        self.create_known_people_tab()

    # ============================================================
    # VIDEO RECOGNITION TAB
    # ============================================================

    def create_video_tab(self):

        title = tk.Label(
            self.video_tab,
            text="Face Recognition",
            font=("Arial", 20, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = tk.Label(
            self.video_tab,
            text="Select a video to detect and recognize people.",
            font=("Arial", 10),
            bg=self.bg_color,
            fg=self.secondary_text
        )

        subtitle.pack(
            pady=(0, 20)
        )

        # --------------------------------
        # Video selection
        # --------------------------------

        video_frame = tk.Frame(
            self.video_tab,
            bg=self.panel_color,
            padx=15,
            pady=15
        )

        video_frame.pack(
            fill=tk.X,
            padx=30,
            pady=10
        )

        self.video_path_label = tk.Label(
            video_frame,
            text="No video selected",
            font=("Arial", 10),
            bg=self.panel_color,
            fg=self.secondary_text,
            anchor="w"
        )

        self.video_path_label.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True
        )

        select_button = tk.Button(
            video_frame,
            text="Select Video",
            command=self.select_video,
            bg=self.button_color,
            fg="white",
            activebackground=self.accent_color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=15,
            pady=7,
            cursor="hand2"
        )

        select_button.pack(
            side=tk.RIGHT
        )

        # --------------------------------
        # Start button
        # --------------------------------

        self.start_button = tk.Button(
            self.video_tab,
            text="Start Recognition",
            command=self.start_recognition,
            bg=self.button_color,
            fg="white",
            activebackground=self.accent_color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            font=("Arial", 11, "bold"),
            cursor="hand2"
        )

        self.start_button.pack(
            pady=15
        )

        # --------------------------------
        # Status
        # --------------------------------

        self.status_label = tk.Label(
            self.video_tab,
            text="Ready",
            font=("Arial", 10),
            bg=self.bg_color,
            fg=self.secondary_text
        )

        self.status_label.pack(
            pady=5
        )

        # --------------------------------
        # Results area
        # --------------------------------

        results_title = tk.Label(
            self.video_tab,
            text="People Seen",
            font=("Arial", 14, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )

        results_title.pack(
            pady=(15, 5)
        )

        self.results_container = tk.Frame(
            self.video_tab,
            bg=self.bg_color
        )

        self.results_container.pack(
            fill=tk.BOTH,
            expand=True,
            padx=30,
            pady=10
        )

        self.results_frame = tk.Frame(
            self.results_container,
            bg=self.bg_color
        )

        self.results_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.video_path = None

    # ============================================================
    # VIDEO FUNCTIONS
    # ============================================================

    def select_video(self):

        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.video_path = path

        self.video_path_label.config(
            text=path,
            fg=self.text_color
        )

        self.status_label.config(
            text="Video selected."
        )

    def start_recognition(self):

        if not self.video_path:

            messagebox.showwarning(
                "No Video",
                "Please select a video first."
            )

            return

        self.start_button.config(
            state=tk.DISABLED
        )

        self.status_label.config(
            text="Processing video..."
        )

        self.clear_results()

        thread = threading.Thread(
            target=self.process_video,
            daemon=True
        )

        thread.start()

    def process_video(self):

        try:

            result = read_video(
                self.video_path
            )

            self.root.after(
                0,
                lambda: self.show_results(result)
            )

        except Exception as error:

            self.root.after(
                0,
                lambda: self.show_error(error)
            )

    def show_error(self, error):

        self.status_label.config(
            text="An error occurred."
        )

        self.start_button.config(
            state=tk.NORMAL
        )

        messagebox.showerror(
            "Error",
            str(error)
        )

    # ============================================================
    # RESULTS
    # ============================================================

    def clear_results(self):

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        self.photo_refs = []

    def show_results(self, result):

        self.clear_results()

        people = result

        if not people:

            label = tk.Label(
                self.results_frame,
                text="No recognized people found.",
                font=("Arial", 12),
                bg=self.bg_color,
                fg=self.secondary_text
            )

            label.pack(
                pady=20
            )

        else:

            for person, face in sorted(
                people.items()
            ):

                self.create_person_result(
                    person,
                    face
                )

        self.status_label.config(
            text="Processing complete!"
        )

        self.start_button.config(
            state=tk.NORMAL
        )

    def create_person_result(
        self,
        person,
        face
    ):

        card = tk.Frame(
            self.results_frame,
            bg=self.card_color,
            bd=1,
            relief=tk.GROOVE,
            padx=10,
            pady=10
        )

        card.pack(
            side=tk.LEFT,
            padx=10,
            pady=10
        )

        # OpenCV BGR -> RGB

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # NumPy -> PIL

        image = Image.fromarray(
            face_rgb
        )

        # Resize

        image.thumbnail(
            (150, 150)
        )

        # PIL -> Tkinter

        photo = ImageTk.PhotoImage(
            image
        )

        # Keep reference alive

        self.photo_refs.append(
            photo
        )

        image_label = tk.Label(
            card,
            image=photo,
            bg=self.card_color
        )

        image_label.pack()

        name_label = tk.Label(
            card,
            text=person,
            font=("Arial", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )

        name_label.pack(
            pady=(8, 0)
        )

    # ============================================================
    # ADD PERSON TAB
    # ============================================================

    def create_add_person_tab(self):

        title = tk.Label(
            self.add_person_tab,
            text="Add Person",
            font=("Arial", 20, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = tk.Label(
            self.add_person_tab,
            text="Register a person using one or more photos.",
            font=("Arial", 10),
            bg=self.bg_color,
            fg=self.secondary_text
        )

        subtitle.pack(
            pady=(0, 20)
        )

        # --------------------------------
        # Name
        # --------------------------------

        name_frame = tk.Frame(
            self.add_person_tab,
            bg=self.bg_color
        )

        name_frame.pack(
            pady=10
        )

        name_label = tk.Label(
            name_frame,
            text="Name:",
            font=("Arial", 11),
            bg=self.bg_color,
            fg=self.text_color
        )

        name_label.pack(
            side=tk.LEFT,
            padx=5
        )

        self.name_entry = tk.Entry(
            name_frame,
            width=35,
            bg=self.panel_color,
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT
        )

        self.name_entry.pack(
            side=tk.LEFT,
            padx=5,
            ipady=6
        )

        # --------------------------------
        # Photo selection
        # --------------------------------

        photo_button = tk.Button(
            self.add_person_tab,
            text="Select Photos",
            command=self.select_photos,
            bg=self.button_color,
            fg="white",
            activebackground=self.accent_color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2"
        )

        photo_button.pack(
            pady=15
        )

        self.photo_label = tk.Label(
            self.add_person_tab,
            text="No photos selected",
            font=("Arial", 10),
            bg=self.bg_color,
            fg=self.secondary_text
        )

        self.photo_label.pack(
            pady=5
        )

        # --------------------------------
        # Add button
        # --------------------------------

        add_button = tk.Button(
            self.add_person_tab,
            text="Add Person",
            command=self.add_person_to_database,
            bg=self.button_color,
            fg="white",
            activebackground=self.accent_color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=10,
            font=("Arial", 11, "bold"),
            cursor="hand2"
        )

        add_button.pack(
            pady=20
        )

        self.selected_photos = []

    def select_photos(self):

        paths = filedialog.askopenfilenames(
            title="Select Photos",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )

        if not paths:
            return

        self.selected_photos = list(
            paths
        )

        self.photo_label.config(
            text=f"{len(paths)} photo(s) selected.",
            fg=self.text_color
        )

    def add_person_to_database(self):

        name = self.name_entry.get().strip()

        if not name:

            messagebox.showwarning(
                "Missing Name",
                "Please enter a person's name."
            )

            return

        if not self.selected_photos:

            messagebox.showwarning(
                "Missing Photos",
                "Please select at least one photo."
            )

            return

        try:

            add_person(name)

            for photo in self.selected_photos:

                add_embedding(
                    name,
                    photo
                )

            messagebox.showinfo(
                "Success",
                f"{name} was added successfully."
            )

            # Clear fields

            self.name_entry.delete(
                0,
                tk.END
            )

            self.selected_photos = []

            self.photo_label.config(
                text="No photos selected.",
                fg=self.secondary_text
            )

            # Refresh Known People

            self.refresh_known_people()

        except Exception as error:

            messagebox.showerror(
                "Error",
                str(error)
            )

    # ============================================================
    # KNOWN PEOPLE TAB
    # ============================================================

    def create_known_people_tab(self):

        title = tk.Label(
            self.known_people_tab,
            text="Known People",
            font=("Arial", 20, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )

        title.pack(
            pady=(20, 5)
        )

        subtitle = tk.Label(
            self.known_people_tab,
            text="People currently registered in the database.",
            font=("Arial", 10),
            bg=self.bg_color,
            fg=self.secondary_text
        )

        subtitle.pack(
            pady=(0, 20)
        )

        self.known_people_frame = tk.Frame(
            self.known_people_tab,
            bg=self.bg_color
        )

        self.known_people_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=30,
            pady=10
        )

        self.refresh_known_people()

    def refresh_known_people(self):

        for widget in self.known_people_frame.winfo_children():
            widget.destroy()

        people = get_known_people()

        if not people:

            label = tk.Label(
                self.known_people_frame,
                text="No known people yet.",
                font=("Arial", 12),
                bg=self.bg_color,
                fg=self.secondary_text
            )

            label.pack(
                pady=30
            )

            return

        for person in people:

            card = tk.Frame(
                self.known_people_frame,
                bg=self.card_color,
                bd=1,
                relief=tk.GROOVE
            )

            card.pack(
                fill=tk.X,
                padx=10,
                pady=5
            )

            name_label = tk.Label(
                card,
                text=person,
                font=("Arial", 12, "bold"),
                bg=self.card_color,
                fg=self.text_color,
                anchor="w"
            )

            name_label.pack(
                fill=tk.X,
                padx=15,
                pady=12
            )


# ================================================================
# START GUI
# ================================================================

def start_gui():

    root = tk.Tk()

    app = FaceRecognitionGUI(
        root
    )

    root.mainloop()

