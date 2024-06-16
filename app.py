import cv2
from cv2 import VideoCapture, waitKey
import numpy as np
from keras.models import load_model
import  keras.utils as image
import telepot
import time
import random
from twilio.rest import Client

# Load the pre-trained Keras model and initialize the webcam
model = load_model("./keras_model.h5")
cap = cv2.VideoCapture(0)

class_labels = ['Humans', 'Accidents', 'No Accidents']

# Initialize your Telegram bot
bot_token = "6458901047:AAFKtOpX_hMpz-yabhOloPHtiMLouCwZHaI"
chat_id = "1465506330"
bot = telepot.Bot(bot_token)
last_detection_time = time.time()

# Initialize your Twilio credentials and phone numbers
twilio_account_sid = "ACf1cba440bb340b9b103fce65d51eaa86"
twilio_auth_token = "9c7274105f9fa8ac83c4024193b5fac9"
twilio_phone_number = "+17047614717"
recipient_phone_number = "+917396198485"

# Create a Twilio client
client = Client(twilio_account_sid, twilio_auth_token)

# List of random demonstration locations (replace with real locations)
demo_locations = [
    "New York, USA",
    "Los Angeles, USA",
    "London, UK",
    "Paris, France",
    "Tokyo, Japan",
]

# Dictionary to keep track of user responses
user_responses = {}

# Function to send an alert to the other chat ID
def send_alert_to_other_chat(location, timestamp):
    bot.sendMessage(chat_id, f"🚨 ALERT: Accident Detected 🚨\n"
                                  f"📅 Timestamp: {timestamp} 🕒\n"
                                  f"At Location: {location}\n"
                                  f"No ambulance needed. Stay safe.")

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    prediction = model.predict(img_tensor)
    class_index = np.argmax(prediction[0])
    class_label = class_labels[class_index]
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow('Webcam', frame)

    if class_label == 'Accidents' and time.time() - last_detection_time >= 5:
        # Generate a random location for demonstration
        random_location = random.choice(demo_locations)

        cv2.imwrite('captured_image.jpg', frame)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open('captured_image.jpg', 'rb') as photo:
            caption = f'''🚨 ALERT: {class_label} Detected 🚨
                        📅 Timestamp: {current_time} 🕒
                        At Location: {random_location}
                        1️⃣ Ambulance on the way.
                        2️⃣ No ambulance needed.'''
            message = bot.sendPhoto(chat_id, photo, caption=caption)

        # Store the message ID in user_responses
        user_responses[message['message_id']] = {
            'timestamp': current_time,
            'location': random_location,
        }

        # Send an SMS using Twilio
        client.messages.create(
            to=recipient_phone_number,
            from_=twilio_phone_number,
            body=f"🚨 ALERT: {class_label} Detected 🚨\n"
                 f"📅 Timestamp: {current_time} 🕒\n"
                 f"At Location: {random_location}\n"
                 f"Please check Telegram for details.")

        last_detection_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Now, you can use a separate function to handle user responses and act accordingly.

# Add your user response handling code here...

while True:
    time.sleep(1)
