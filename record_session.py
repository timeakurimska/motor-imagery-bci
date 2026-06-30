import random
import threading
import tkinter as tk
from PIL import Image, ImageTk

from crown_handler import CrownHandler


X_test = ["./rest.png", "./left.png", "./right.png"]
y_test = [0, 1, 2]

N_TIMES = 100
IMG_SHOW_TIME = 6

N_TIMES = 100
IMG_SHOW_TIME = 6
SESSION_NAME = "s08" 


def generate_no_consecutive_sequence(n, items):
    sequence = []
    prev = None

    for _ in range(n):
        candidates = [x for x in items if x != prev]
        choice = random.choice(candidates)

        sequence.append(choice)
        prev = choice

    return sequence


indices = generate_no_consecutive_sequence(N_TIMES, [0, 1, 2])
image_sequence = [(X_test[i], y_test[i]) for i in indices]


crown = CrownHandler()
data_lock = threading.Lock()


def crown_callback(data):
    with data_lock:
        crown.add_sample(data)


def crown_run_display():

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.configure(bg="black")

    canvas = tk.Canvas(root, bg="black", highlightthickness=0)
    canvas.pack(expand=True, fill="both")

    PROGRESS_WIDTH = 400
    PROGRESS_HEIGHT = 14
    PROGRESS_TOP_PADDING = 40

    left_edge = (root.winfo_screenwidth() - PROGRESS_WIDTH) // 2

    progress_bg = canvas.create_rectangle(
        left_edge,
        PROGRESS_TOP_PADDING,
        left_edge + PROGRESS_WIDTH,
        PROGRESS_TOP_PADDING + PROGRESS_HEIGHT,
        fill="#333333",
        outline="",
        tags="progress"
    )

    progress_fill = canvas.create_rectangle(
        left_edge,
        PROGRESS_TOP_PADDING,
        left_edge,
        PROGRESS_TOP_PADDING + PROGRESS_HEIGHT,
        fill="#4CAF50",
        outline="",
        tags="progress"
    )

    def update_progress(idx):

        if idx < 0 or len(image_sequence) == 0:
            return

        fraction = (idx + 1) / len(image_sequence)
        filled_width = PROGRESS_WIDTH * fraction

        canvas.coords(
            progress_fill,
            left_edge,
            PROGRESS_TOP_PADDING,
            left_edge + filled_width,
            PROGRESS_TOP_PADDING + PROGRESS_HEIGHT
        )

    def show_image(idx):

        if idx >= len(image_sequence):
            with data_lock:
                crown.current_label = None

            print("Sequence finished.")
            root.after(800, root.quit)
            return

        update_progress(idx)

        canvas.delete("image")

        path, lab = image_sequence[idx]

        try:

            img = Image.open(path).resize(
                (root.winfo_screenwidth(), root.winfo_screenheight()),
                Image.Resampling.LANCZOS
            )

            imgtk = ImageTk.PhotoImage(img)

            canvas.create_image(
                0,
                0,
                image=imgtk,
                anchor="nw",
                tags="image"
            )

            canvas.image = imgtk

        except Exception as e:

            print(f"Error loading {path}: {e}")

            canvas.create_text(
                root.winfo_screenwidth() // 2,
                root.winfo_screenheight() // 2,
                text="IMAGE ERROR",
                fill="red",
                font=("Arial", 72),
                tags="image"
            )

        with data_lock:
            crown.current_label = lab

        canvas.tag_raise("progress")

        root.after(IMG_SHOW_TIME * 1000, lambda: show_image(idx + 1))

    show_image(0)
    root.mainloop()


if __name__ == "__main__":

    try:

        crown.start_stream(crown_callback)

        crown.prevent_sleep()

        print("→ Starting stimulus presentation")

        crown_run_display()

        print("→ Display finished")

        crown.stop_and_save(N_TIMES, IMG_SHOW_TIME, SESSION_NAME)

    except Exception as e:

        print(f"CRITICAL ERROR: {e}")

    finally:

        crown.allow_sleep()
        print("Experiment finished.")