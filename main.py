from gpiozero import Button
from picamera import PiCamera
from datetime import datetime
from signal import pause
import os
import lcd_lpr
import time

def process():
	button = Button(24)
	camera = PiCamera()
	#led = LED(23)
	def capture():
	    print('Image Recieved!')
	    camera.capture('/home/pi/img.jpg')
            lcd_lpr.mn('img.jpg')


	button.when_pressed = capture
	#button.when_pressed = led.on
	#button.when_released = led.off
	lcd_lpr.setText("Press!")
	pause()
while(True):
    process()