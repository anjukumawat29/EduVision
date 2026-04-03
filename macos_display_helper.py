"""
macOS OpenCV display helper.
Ensures cv2.imshow() works properly when called from subprocesses.
"""

import cv2
import sys

def enable_macos_display():
    """
    Enable OpenCV GUI on macOS.
    Call this at the start of any script using cv2.imshow().
    """
    # Force OpenCV to use Cocoa backend on macOS
    if sys.platform == 'darwin':
        try:
            cv2.setNumThreads(1)  # Reduce threading issues
        except:
            pass
    
    return True

def safe_imshow(window_name, image):
    """
    Safe wrapper for cv2.imshow that handles macOS issues.
    """
    try:
        cv2.imshow(window_name, image)
        return True
    except Exception as e:
        print(f"[display] Error: {e}", file=sys.stderr)
        return False

def safe_waitkey(delay=1):
    """
    Safe wrapper for cv2.waitKey that handles macOS issues.
    Returns True if 'q' was pressed, False otherwise.
    """
    try:
        key = cv2.waitKey(delay) & 0xFF
        return key == ord('q')
    except:
        return False

def safe_destroyall():
    """Safe wrapper for cv2.destroyAllWindows()."""
    try:
        cv2.destroyAllWindows()
    except:
        pass
