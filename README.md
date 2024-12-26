## Interactive-Border-Detection

The code base provide real-time border and color detection and save frames for post process detection. 
To use the app, run camera.py, and 4 video stream will show. 
- top left is gray-sacle image
- top right is the sobel filter result
- bottom left is the color mask result
- bottom right is the high pass filter result

The frames taken is saved to the images/ folder. To detect, or place mark at ground border, run save_ground.py, and the result will be saved into ground/ folder. To detect windows, run save_output.py, and the result will be saved into window/ folder.
