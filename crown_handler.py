import os
import json
import time
import ctypes
import platform
import subprocess

from neurosity import NeurositySDK
from dotenv import load_dotenv


class CrownHandler:
    def __init__(self):
        load_dotenv()

        self.email = os.getenv("NEUROSITY_EMAIL")
        self.password = os.getenv("NEUROSITY_PASSWORD")
        self.device_id = os.getenv("NEUROSITY_DEVICE_ID")

        if not all([self.email, self.password, self.device_id]):
            raise ValueError("Missing Neurosity credentials in .env file")

        self.neurosity = NeurositySDK({"device_id": self.device_id})
        self._login()

        self.data_list = []
        self.current_label = None
        self.unsubscribe_func = None

    def _login(self):
        """Login to Neurosity account"""
        try:
            self.neurosity.login({
                "email": self.email,
                "password": self.password
            })
            print("→ Neurosity login successful")
        except Exception as e:
            raise ConnectionError(f"Neurosity login failed: {e}")

    def prevent_sleep(self):
        """Prevent the system from sleeping during the experiment"""
        system = platform.system()

        if system == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

        elif system == "Darwin":
            subprocess.Popen(['caffeinate', '-di'])

        elif system == "Linux":
            os.system('xset s off -dpms')

    def start_stream(self, callback):
        """Start streaming raw brainwaves"""
        self.unsubscribe_func = self.neurosity.brainwaves_raw(callback)
        print("→ Brainwaves stream started")

    def stop_stream(self):
        """Stop the brainwaves stream"""
        if callable(self.unsubscribe_func):
            try:
                self.unsubscribe_func()
                print("→ Stream stopped")
            except Exception as e:
                print(f"Error stopping stream: {e}")

            self.unsubscribe_func = None

    def add_sample(self, data):
        """Attach label to EEG sample"""
        if self.current_label is not None:
            marked = data.copy()
            marked["mark"] = self.current_label
            self.data_list.append(marked)

    
    def stop_and_save(self, n_times: int, show_time_sec: int, session_name: str):
        """Stop stream and save data"""
        self.stop_stream()

        # wait for last packets
        time.sleep(0.7)

        os.makedirs("data", exist_ok=True)

        filename = f"data/brainwaves_{session_name}_N{n_times}_T{show_time_sec}s.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.data_list, f, indent=2, ensure_ascii=False)

        print(f"→ Data saved: {filename} ({len(self.data_list)} samples)")


    def allow_sleep(self):
        system = platform.system()
        if system == "Darwin":
            os.system('killall caffeinate')
        elif system == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)