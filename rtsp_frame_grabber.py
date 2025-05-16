import cv2
import threading
import time
import queue

class RTSPFrameGrabber:
    def __init__(self, rtsp_url, buffer_size=10, reconnect_attempts=5, reconnect_timeout=3):
        """
        A specialized frame grabber for RTSP streams with reconnection and buffering capabilities.
        
        Args:
            rtsp_url (str): The RTSP URL to connect to
            buffer_size (int): Maximum number of frames to buffer
            reconnect_attempts (int): Number of reconnection attempts before giving up
            reconnect_timeout (int): Seconds to wait between reconnection attempts
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_timeout = reconnect_timeout
        
        # Create frame buffer queue
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.last_frame = None
        
        # Stream properties
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        
        # Control flags
        self.is_running = False
        self.is_connected = False
        self.frame_available = False
        
        # Thread for grabbing frames
        self.grab_thread = None
        
        # Stats
        self.dropped_frames = 0
        self.total_frames = 0
        self.reconnect_count = 0
        self.last_frame_time = 0
        
        # Connect when initialized
        self.connect()
    
    def connect(self):
        """Connect to the RTSP stream and configure the capture"""
        print(f"Connecting to RTSP stream: {self.rtsp_url}")
        
        # Configure capture with RTSP optimizations
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimize for RTSP streaming
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer for lower latency
        
        # Check if connection was successful
        if not self.cap.isOpened():
            print(f"Failed to connect to RTSP stream: {self.rtsp_url}")
            self.is_connected = False
            return False
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Connected to RTSP stream: {self.width}x{self.height} @ {self.fps} FPS")
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from the RTSP stream"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("Disconnected from RTSP stream")
    
    def start(self):
        """Start the frame grabbing thread"""
        if self.is_running:
            print("Frame grabber is already running")
            return
        
        if not self.is_connected:
            if not self.connect():
                print("Cannot start frame grabber: not connected to stream")
                return
        
        # Clear the buffer
        while not self.frame_buffer.empty():
            self.frame_buffer.get()
        
        # Start the grabbing thread
        self.is_running = True
        self.grab_thread = threading.Thread(target=self._grab_frames)
        self.grab_thread.daemon = True
        self.grab_thread.start()
        print("Frame grabber started")
    
    def stop(self):
        """Stop the frame grabbing thread"""
        self.is_running = False
        if self.grab_thread is not None:
            self.grab_thread.join(timeout=1.0)
            self.grab_thread = None
        self.disconnect()
        print("Frame grabber stopped")
    
    def _grab_frames(self):
        """Thread function to continuously grab frames"""
        reconnect_counter = 0
        
        while self.is_running:
            if not self.is_connected:
                reconnect_counter += 1
                if reconnect_counter > self.reconnect_attempts:
                    print(f"Failed to reconnect after {self.reconnect_attempts} attempts")
                    self.is_running = False
                    break
                
                print(f"Attempting to reconnect ({reconnect_counter}/{self.reconnect_attempts})...")
                if self.connect():
                    reconnect_counter = 0
                    self.reconnect_count += 1
                else:
                    time.sleep(self.reconnect_timeout)
                    continue
            
            # Try to grab a frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame, stream may be disconnected")
                self.is_connected = False
                self.frame_available = False
                time.sleep(0.1)
                continue
            
            self.total_frames += 1
            self.last_frame_time = time.time()
            
            # Add frame to buffer, drop oldest if full
            try:
                if self.frame_buffer.full():
                    self.frame_buffer.get_nowait()  # Drop oldest frame                   
                    self.dropped_frames += 1
                    
                self.last_frame = frame
                self.frame_buffer.put_nowait(frame)
                self.frame_available = True
            except queue.Full:
                self.dropped_frames += 1
                time.sleep(0.001)  # Small sleep to prevent CPU hogging
    
    def read(self):
        """Read a frame from the buffer, similar to cv2.VideoCapture.read()"""
        if not self.is_connected:
            return False, None
        
        if self.frame_buffer.empty():
            return False, None
        
        try:
            
            # frame = self.frame_buffer.get_nowait()
            frame = self.last_frame.copy()  # Return a copy of the last frame

            return True, frame
        except queue.Empty:
            return False, None
    
    def get_stream_info(self):
        """Get information about the stream and grabber status"""
        buffer_usage = self.frame_buffer.qsize() / self.buffer_size if self.buffer_size > 0 else 0
        
        return {
            "connected": self.is_connected,
            "running": self.is_running,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "buffer_usage": buffer_usage,
            "reconnect_count": self.reconnect_count,
            "last_frame_time": self.last_frame_time
        }
    
    def isOpened(self):
        """Mimic the cv2.VideoCapture.isOpened() method"""
        return self.is_connected
