from gpiozero import Button
from picamera import PiCamera
from datetime import datetime
from signal import pause
import os


def process():
	button = Button(2)
	camera = PiCamera()

	def capture():
	    camera.capture('/home/pi/img.jpeg')

	button.when_pressed = capture
	os.system('python lcd_lpr.py img.jpg')

	pause()


while("pigs" != "fly"):
	process()