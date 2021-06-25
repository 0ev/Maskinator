from gtts import gTTS
from playsound import playsound
import threading
import os
import time

class Speaker:
    def __init__(self):
        self.is_speaking = False
        self.thread = threading.Thread(target = self.mypass)
        self.thread.start()
    
    def mypass(self):
        pass
    
    def speak(self,color):
        if not self.thread.is_alive():
            self.thread = threading.Thread(target = self._speak, args = (color,))
            self.thread.start()

    def _speak(self,color):
        if color != None:
            text = f"그 {color} 옷 입은 사람 마스크 쓰세요"
            tts = gTTS(text=text, lang='ko')
            tts.save("speach.mp3")
            time.sleep(0.1)
            playsound('speach.mp3')
            os.remove("speach.mp3")

if __name__ == "__main__":
    a = Speaker()
    a.speak("하얀색")