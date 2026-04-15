import time
try:
    from pynput.keyboard import Key, Controller
except ImportError:
    print("Please install pynput: pip install pynput")
    exit(1)

keyboard = Controller()

# Map original Windows DirectInput scan codes to Linux pynput keys
# Right arrow (0x4D in Windows) -> Key.right
# Left arrow (0x4B in Windows) -> Key.left
key_map = {
    0x4D: Key.right,
    0x4B: Key.left,
    0x11: 'w'   # 0x11 mapping for the test case example
}

# Keep variable names from the original script to avoid breaking imports
right_pressed = 0x4D
left_pressed = 0x4B

def PressKey(hexKeyCode):
    """Simulate a key-down event."""
    key = key_map.get(hexKeyCode)
    if key:
        keyboard.press(key)

def ReleaseKey(hexKeyCode):
    """Simulate a key-up event."""
    key = key_map.get(hexKeyCode)
    if key:
        keyboard.release(key)

if __name__ == '__main__':
    # Tiny local test: press and release mapping for 0x11 repeatedly once per second.
    while True:
        PressKey(0x11)
        time.sleep(1)
        ReleaseKey(0x11)
        time.sleep(1)
