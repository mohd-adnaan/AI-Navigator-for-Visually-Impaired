import speech_recognition as sr
import threading
import os
import pyttsx3

speak=False
keyword = ''

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# recognize speech using Sphinx
try:
    keyword = r.recognize_sphinx(audio)
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    if keyword != '':
        speak = True	
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))

def sayItem():
    global keyword
    global speak
    while True:
        if speak ==True:
            engine = pyttsx3.init()
            engine.setProperty('rate',150)
            text = " What's up bitch?"
            engine.say(text)
            engine.runAndWait()
            speak=False


y=threading.Thread(target=sayItem, daemon=True)
y.start()
