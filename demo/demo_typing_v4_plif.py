"""Real-time BCI Typing Demo - V5-PLIF (1s, Best Accuracy: 84.17%)
========================================
Type a message on your physical keyboard, and the model will "read your mind"
using real EEG trials from S35. Compare what you typed vs what the model predicted.

Usage:
    python demo/demo_typing_v4_plif.py

Features:
- Physical keyboard input (type naturally, no mouse clicking)
- Real-time character-by-character prediction using S35 EEG data
- Visual comparison: target text vs predicted text (green=correct, red=wrong)
- Live virtual keyboard with highlight feedback
- Typing speed and accuracy stats
- Modern dark UI theme
"""
import os
import sys
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, font as tkfont
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.model_v4_plif import FBSpikeSSVEPformerV4PLIF, functional

# Map keyboard keys to class IDs (0-39)
KEY_MAP = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
    'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
    'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25,
    '1': 26, '2': 27, '3': 28, '4': 29, '5': 30, '6': 31, '7': 32, '8': 33, '9': 34, '0': 35,
    '.': 36, ',': 37, '!': 38, '?': 39,
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
}

CLASS_LABELS = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','1','2','3','4',
    '5','6','7','8','9','0','.',',','!','?'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'results_v4_plif_fixed_v5/v4_plif_S35_model.pth'
CACHE_DIR = 'cache_1s'


class BCITypingDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("BCI Brain-Typer - V5-PLIF (1s, Best: 84.17%)")
        self.root.geometry("1100x800")
        self.root.configure(bg='#1a1a2e')
        self.root.minsize(900, 650)
        
        # Load model and data
        self.model = self.load_model()
        self.X_test, self.y_test = self.load_data()
        self.class_trials = self.group_by_class()
        self.used_trials = {c: set() for c in range(40)}
        
        # State
        self.target_text = ""
        self.pred_text = ""
        self.correct_count = 0
        self.total_count = 0
        self.is_running = False
        self.current_idx = 0
        self.typing_delay = 0.05  # seconds between characters
        
        # Fonts
        self.font_large = tkfont.Font(family="Consolas", size=32, weight="bold")
        self.font_medium = tkfont.Font(family="Consolas", size=18)
        self.font_small = tkfont.Font(family="Consolas", size=14)
        self.font_keyboard = tkfont.Font(family="Arial", size=20, weight="bold")
        
        self.setup_ui()
        # Bind only Return to start the demo; Entry widget handles all typing natively
        self.input_entry.bind('<Return>', self.on_start_typing)
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = tk.Label(main_frame, text="🧠 BCI Brain-Typer", 
                         font=tkfont.Font(family="Arial", size=28, weight="bold"),
                         bg='#1a1a2e', fg='#e94560')
        title.pack(pady=(0, 5))
        
        subtitle = tk.Label(main_frame, text="V5-PLIF | 1s Window | 40-Class SSVEP | Best: 84.17%",
                           font=self.font_small, bg='#1a1a2e', fg='#a0a0a0')
        subtitle.pack(pady=(0, 15))
        
        # Input area
        input_frame = tk.Frame(main_frame, bg='#16213e', bd=2, relief=tk.RIDGE)
        input_frame.pack(fill=tk.X, pady=10, ipady=10)
        
        input_label = tk.Label(input_frame, text="Type your message:",
                              font=self.font_medium, bg='#16213e', fg='#e0e0e0')
        input_label.pack(side=tk.LEFT, padx=15)
        
        self.input_entry = tk.Entry(input_frame, font=self.font_medium, width=40,
                                    bg='#0f3460', fg='white', insertbackground='white',
                                    bd=2, relief=tk.FLAT)
        self.input_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        self.input_entry.focus_set()
        
        self.btn_start = tk.Button(input_frame, text="▶ Start", font=self.font_medium,
                                   bg='#e94560', fg='white', activebackground='#ff6b81',
                                   bd=0, padx=20, pady=5, cursor='hand2',
                                   command=self.start_typing_thread)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_clear = tk.Button(input_frame, text="✕ Clear", font=self.font_medium,
                                   bg='#533483', fg='white', activebackground='#6a4c93',
                                   bd=0, padx=20, pady=5, cursor='hand2',
                                   command=self.clear_all)
        self.btn_clear.pack(side=tk.LEFT, padx=5)
        
        # Result display area
        result_frame = tk.Frame(main_frame, bg='#1a1a2e')
        result_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        # Target row
        target_frame = tk.Frame(result_frame, bg='#1a1a2e')
        target_frame.pack(fill=tk.X, pady=5)
        tk.Label(target_frame, text="TARGET:", font=self.font_medium, 
                bg='#1a1a2e', fg='#00d2ff', width=10, anchor='e').pack(side=tk.LEFT)
        self.target_display = tk.Label(target_frame, text="", font=self.font_large,
                                      bg='#1a1a2e', fg='#00d2ff', anchor='w')
        self.target_display.pack(side=tk.LEFT, padx=10)
        
        # Predicted row
        pred_frame = tk.Frame(result_frame, bg='#1a1a2e')
        pred_frame.pack(fill=tk.X, pady=5)
        tk.Label(pred_frame, text="PRED:  ", font=self.font_medium,
                bg='#1a1a2e', fg='#ff6b6b', width=10, anchor='e').pack(side=tk.LEFT)
        self.pred_display = tk.Label(pred_frame, text="", font=self.font_large,
                                    bg='#1a1a2e', fg='white', anchor='w')
        self.pred_display.pack(side=tk.LEFT, padx=10)
        
        # Accuracy bar
        self.acc_bar = tk.Canvas(result_frame, bg='#0f3460', height=30, 
                                 highlightthickness=0)
        self.acc_bar.pack(fill=tk.X, pady=10, padx=10)
        self.acc_bar.create_rectangle(0, 0, 0, 30, fill='#e94560', tags='bar')
        self.acc_bar.create_text(450, 15, text="Accuracy: 0.0%", fill='white',
                                 font=self.font_medium, tags='text')
        
        # Stats
        stats_frame = tk.Frame(result_frame, bg='#1a1a2e')
        stats_frame.pack(fill=tk.X, pady=5)
        self.stats_label = tk.Label(stats_frame, text="Correct: 0/0 | ITR: 0.0 bits/min | Speed: 0.0s/char",
                                    font=self.font_small, bg='#1a1a2e', fg='#a0a0a0')
        self.stats_label.pack()
        
        # Virtual Keyboard
        kb_frame = tk.Frame(main_frame, bg='#1a1a2e', bd=2, relief=tk.RIDGE)
        kb_frame.pack(fill=tk.X, pady=10, ipady=10)
        
        tk.Label(kb_frame, text="Virtual Keyboard", font=self.font_medium,
                bg='#1a1a2e', fg='#e0e0e0').pack(pady=5)
        
        self.key_buttons = {}
        for row_idx, row in enumerate([
            ['A','B','C','D','E','F','G','H','I','J'],
            ['K','L','M','N','O','P','Q','R','S','T'],
            ['U','V','W','X','Y','Z','1','2','3','4'],
            ['5','6','7','8','9','0','.',',','!','?']
        ]):
            row_frame = tk.Frame(kb_frame, bg='#1a1a2e')
            row_frame.pack(pady=3)
            for col_idx, ch in enumerate(row):
                class_id = row_idx * 10 + col_idx
                btn = tk.Label(row_frame, text=ch, font=self.font_keyboard,
                              width=3, height=1, bg='#16213e', fg='#e0e0e0',
                              bd=2, relief=tk.RAISED)
                btn.pack(side=tk.LEFT, padx=3)
                self.key_buttons[class_id] = btn
        
        # Status bar
        self.status_bar = tk.Label(main_frame, text="Ready. Type a message and press Enter or click Start.",
                                   font=self.font_small, bg='#0f3460', fg='white',
                                   bd=1, relief=tk.SUNKEN, anchor='w')
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
    def load_model(self):
        print("Loading V5 model (best accuracy: 84.17%)...")
        model = FBSpikeSSVEPformerV4PLIF(
            Chans=11, n_classes=40, fs=250,
            band=[8, 45], resolution=0.25, drop_rate=0.5,
            n_subbands=3, T_snn=12
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("V5 model loaded.")
        return model
    
    def load_data(self):
        cache_dir = os.path.join('..', CACHE_DIR) if not os.path.exists(CACHE_DIR) else CACHE_DIR
        X = np.load(os.path.join(cache_dir, 'S35_data.npy')).astype(np.float32)
        y = np.load(os.path.join(cache_dir, 'S35_labels.npy')).astype(np.int64)
        return X, y
    
    def group_by_class(self):
        d = {c: [] for c in range(40)}
        for i, label in enumerate(self.y_test):
            d[label].append(i)
        return d
    
    def get_trial(self, class_id):
        available = [i for i in self.class_trials[class_id] if i not in self.used_trials[class_id]]
        if not available:
            self.used_trials[class_id] = set()
            available = self.class_trials[class_id]
        idx = np.random.choice(available)
        self.used_trials[class_id].add(idx)
        return self.X_test[idx:idx+1]
    
    def predict(self, x):
        with torch.no_grad():
            x_t = torch.from_numpy(x).to(DEVICE)
            functional.reset_net(self.model)
            out = self.model(x_t)
            pred = out.argmax(dim=1).cpu().item()
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        return pred, probs[pred]
    
    def on_start_typing(self, event=None):
        self.start_typing_thread()
        return 'break'
    
    def start_typing_thread(self):
        text = self.input_entry.get().strip().upper()
        if not text:
            self.status_bar.config(text="Please type a message first!")
            return
        self.target_text = text
        self.pred_text = ""
        self.correct_count = 0
        self.total_count = 0
        self.current_idx = 0
        self.is_running = True
        self.input_entry.config(state='disabled')
        self.btn_start.config(state='disabled', text="Running...")
        
        thread = threading.Thread(target=self.typing_loop, daemon=True)
        thread.start()
    
    def typing_loop(self):
        start_time = time.time()
        
        for i, char in enumerate(self.target_text):
            if char == ' ':
                self.pred_text += ' '
                self.current_idx = i + 1
                self.update_display()
                self.root.after(0, lambda: self.status_bar.config(text=f"Space skipped"))
                time.sleep(self.typing_delay / 2)
                continue
            
            class_id = KEY_MAP.get(char.lower(), None)
            if class_id is None:
                self.pred_text += '?'
                self.current_idx = i + 1
                self.update_display()
                continue
            
            # Get EEG and predict
            x = self.get_trial(class_id)
            pred_id, conf = self.predict(x)
            pred_char = CLASS_LABELS[pred_id]
            
            self.pred_text += pred_char
            self.total_count += 1
            if pred_id == class_id:
                self.correct_count += 1
            
            self.current_idx = i + 1
            
            # Update UI in main thread
            self.root.after(0, self.update_display)
            self.root.after(0, lambda c=class_id, p=pred_id, conf=conf: 
                self.highlight_keyboard(c, p, conf))
            self.root.after(0, lambda char=char, pred=pred_char: 
                self.status_bar.config(text=f"Target: {char} | Predicted: {pred} | Confidence: {conf*100:.1f}%"))
            
            time.sleep(self.typing_delay)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / max(self.total_count, 1)
        
        self.root.after(0, lambda: self.status_bar.config(
            text=f"Done! Accuracy: {self.correct_count}/{self.total_count} = {self.correct_count/max(self.total_count,1)*100:.1f}%"
        ))
        self.root.after(0, lambda: self.input_entry.config(state='normal'))
        self.root.after(0, lambda: self.btn_start.config(state='normal', text="▶ Start"))
        self.is_running = False
    
    def update_display(self):
        # Build colored text
        target_str = self.target_text[:self.current_idx]
        pred_str = self.pred_text
        
        self.target_display.config(text=target_str)
        self.pred_display.config(text=pred_str)
        
        # Update accuracy bar
        acc = self.correct_count / max(self.total_count, 1) * 100
        bar_width = int(acc * 9)  # scale to canvas width
        self.acc_bar.coords('bar', 0, 0, bar_width, 30)
        self.acc_bar.itemconfig('text', text=f"Accuracy: {acc:.1f}% ({self.correct_count}/{self.total_count})")
        
        # Color the bar based on accuracy
        if acc >= 80:
            color = '#2ecc71'  # green
        elif acc >= 60:
            color = '#f1c40f'  # yellow
        else:
            color = '#e94560'  # red
        self.acc_bar.itemconfig('bar', fill=color)
        
        # Update stats
        itr = self.compute_itr(acc / 100)
        self.stats_label.config(text=f"Correct: {self.correct_count}/{self.total_count} | "
                                     f"ITR: {itr:.1f} bits/min | "
                                     f"Speed: {self.typing_delay:.1f}s/char")
    
    def highlight_keyboard(self, target_id, pred_id, conf):
        # Reset all keys
        for cid, btn in self.key_buttons.items():
            btn.config(bg='#16213e', fg='#e0e0e0', bd=2, relief=tk.RAISED)
        
        # Highlight target
        if target_id in self.key_buttons:
            if pred_id == target_id:
                self.key_buttons[target_id].config(bg='#2ecc71', fg='black', bd=3, relief=tk.SUNKEN)
            else:
                self.key_buttons[target_id].config(bg='#e94560', fg='white', bd=3, relief=tk.SUNKEN)
        
        # Highlight prediction if wrong
        if pred_id != target_id and pred_id in self.key_buttons:
            self.key_buttons[pred_id].config(bg='#f1c40f', fg='black', bd=3, relief=tk.SUNKEN)
    
    def compute_itr(self, accuracy):
        if accuracy <= 0 or accuracy >= 1.0:
            return 0.0
        T = 1.0 + 0.5
        return (60.0 / T) * (np.log2(40) + accuracy * np.log2(accuracy) + 
                            (1 - accuracy) * np.log2((1 - accuracy) / 39))
    
    def clear_all(self):
        self.input_entry.config(state='normal')
        self.input_entry.delete(0, tk.END)
        self.target_text = ""
        self.pred_text = ""
        self.correct_count = 0
        self.total_count = 0
        self.current_idx = 0
        self.is_running = False
        self.target_display.config(text="")
        self.pred_display.config(text="")
        self.acc_bar.coords('bar', 0, 0, 0, 30)
        self.acc_bar.itemconfig('text', text="Accuracy: 0.0%")
        self.acc_bar.itemconfig('bar', fill='#e94560')
        self.stats_label.config(text="Correct: 0/0 | ITR: 0.0 bits/min | Speed: 0.0s/char")
        self.status_bar.config(text="Ready. Type a message and press Enter or click Start.")
        for btn in self.key_buttons.values():
            btn.config(bg='#16213e', fg='#e0e0e0', bd=2, relief=tk.RAISED)
        self.btn_start.config(state='normal', text="▶ Start")
        self.input_entry.focus_set()


def main():
    print("=" * 60)
    print("BCI Brain-Typer Demo - V5-PLIF (1s, Best Accuracy: 84.17%)")
    print("Type a message and press Enter to see the model predict!")
    print("=" * 60)
    
    root = tk.Tk()
    app = BCITypingDemo(root)
    root.mainloop()


if __name__ == '__main__':
    main()
