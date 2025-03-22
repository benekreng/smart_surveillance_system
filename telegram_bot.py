import requests
import threading
import json
import time
from datetime import datetime

class TelegramBot:
    def __init__(self):
        token = "PUT_YOUR_TOKEN_HERE"
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.chat_id = "1609932328"
        # True = strict
        self.eval_mode = None
        self.web_app_url = "http://192.168.2.215:8000/"
    
    def evaluation_preference(self, preference):
        if preference:
            self.eval_mode = 'strict'
        else: 
            self.eval_mode = 'relaxed'

    # Sends message using an unblocking thread
    def send_message(self, message):
        thread = threading.Thread(target=self._send_message_thread, args=(message,))
        thread.daemon = True
        thread.start()
    
    def send_alert(self, identities, strict_eval, relaxed_eval):
        # strict_eval = False = not safe
        # In relaxed mode send notification only if relaxed_eval is false
        if self.eval_mode == 'relaxed':
            if relaxed_eval:
                return
        # Only send notification if either condition is false
        else:
            if strict_eval and relaxed_eval:
                return

        # Generate message
        identity_text = "None"
        if identities and len(identities) > 0:
            identity_text = ""
            for identity in identities:
                conf_percent = round(identity.get("confidence", 0) * 100)
                identity_text += f"{identity.get('id')}: {conf_percent}% confidence \n"
        
        # Build message
        spacer = "\n\n"
        date_and_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            #f"Strict Eval, Relaxed Eval: {strict_eval, relaxed_eval}"
            f"<b>Notable Event Detected!</b>{spacer}"
            f"<b>Identified Individuals:</b>\n{identity_text}{spacer}"
            f"<b>At: </b>{date_and_time}{spacer}"
            f'<b>See the Details:</b><a href="{self.web_app_url}">Click here</a>'
        )
        self.send_message(message)
        return message

    
    # Send a message to the user using telegram api
    def _send_message_thread(self, message=" "):
        url = self.base_url + "sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
            "disable_notification": False,
        }
        headers = {
            "accept": "application/json",
            "User-Agent": "Telegram Bot SDK - (https://github.com/irazasyed/telegram-bot-sdk)",
            "content-type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print(f"Error: {response.status_code}, {response.text}")
        

# if __name__ == "__main__":
#     bot = TelegramBot()
#     strict_eval = False
#     relaxed_eval = True
#     identities = [
#         {"id": "Chris", "confidence": 0.95},
#         {"id": "Ben", "confidence": 0.88}
#     ]
#     bot.send_alert(None, strict_eval, relaxed_eval)
#     time.sleep(2)