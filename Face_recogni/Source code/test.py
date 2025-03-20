import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import os
from PIL import Image, ImageTk
import numpy as np
import json


# Create a custom dialog for user input
class CustomDialog(simpledialog.Dialog):
    def body(self, master):
        # Creating labels and entry widgets for ID and Name
        tk.Label(master, text='ID:').grid(row=0)
        tk.Label(master, text='Name:').grid(row=1)

        self.id_entry = tk.Entry(master)
        self.name_entry = tk.Entry(master)

        self.id_entry.grid(row=0, column=1)
        self.name_entry.grid(row=1, column=1)
        return self.id_entry

    def apply(self):
        # Retrieve the user input when the dialog is confirmed
        self.result = (self.id_entry.get(), self.name_entry.get())


# Initialize ID and name lists
id_list = []
name_list = []


# Save lists to JSON file
def save_data():
    with open('data.json', 'w') as file:
        json.dump({'id_list': id_list, 'name_list': name_list}, file)


# Load the data JSON file
def load_data():
    global id_list, name_list
    if os.path.exists('data.json'):
        with open('data.json', 'r') as file:
            data = json.load(file)
            id_list = data.get('id_list', [])
            name_list = data.get('name_list', [])


# Function to collect data and train the model
def collect_and_train():
    # Show custom dialog to get user ID and Name
    input_data = CustomDialog(root)
    if input_data.result is None:
        return
    user_id, user_name = input_data.result

    # Validate the input data
    if not user_id or not user_name:
        tk.messagebox.showerror('Error', 'ID and Name must not be empty.')
        return

    if user_id in id_list:
        tk.messagebox.showerror('Error', 'ID already exists.')
        return

    # Append the ID and Name to the respective lists
    id_list.append(user_id)
    name_list.append(user_name)

    save_data()

    notice_label.config(text='Waitting to get face information...')
    root.update()

    # Initialize video capture and face detector
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(
        r'C:\Users\ADMIN\OneDrive\Desktop\AI1805_G11\DEMO CODE\haarcascade_frontalface_default.xml')

    count = 0

    while True:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            face = gray_frame[y:y + h, x:x + w]
            # Save the captured face images to the directory
            cv2.imwrite(f'collect_data/{user_id}.{id_list.index(user_id)}.{count}.jpg', face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

        cv2.imshow('Collecting Face Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 500:
            break

    video.release()
    cv2.destroyAllWindows()

    # Initialize the face recognizer and prepare for training
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'collect_data'

    def get_images_and_ids(path):
        # Function to retrieve images and their corresponding IDs
        image_paths = [os.path.join(path, i) for i in os.listdir(path)]
        faces = []
        ids = []
        for image in image_paths:
            face_image = Image.open(image).convert('L')
            face_arr = np.array(face_image, 'uint8')
            id = int(os.path.split(image)[-1].split('.')[1])
            faces.append(face_arr)
            ids.append(id)
        return ids, faces

    # Train the recognizer with the collected face data
    ids, faces = get_images_and_ids(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('Trainer.yml')
    notice_label.config(text='')
    root.update()
    messagebox.showinfo('Info', 'Collecting data complete!')


# Function to recognize faces
def recognize_faces():
    video = cv2.VideoCapture(0)
    face_detect = cv2.CascadeClassifier(
        r'C:\Users\ADMIN\OneDrive\Desktop\AI1805_G11\DEMO CODE\haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('Trainer.yml')

    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray_frame[y:y + h, x:x + w]
            id, conf = recognizer.predict(face)
            if conf < 50:
                name = name_list[id]
                confidence = round(conf, 2)
                color = (50, 255, 50)
            else:
                name = 'Unknown'
                confidence = round(conf, 2)
                color = (50, 50, 255)

            cv2.putText(frame, f'{name}   {confidence}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# GUI function
def gui():
    global root, notice_label

    # Main application window
    root = tk.Tk()
    root.title('Face Recognition')

    # Set background image
    background_image = Image.open('background_image.jpg')
    background_photo = ImageTk.PhotoImage(background_image)

    canvas = tk.Canvas(root, width=background_image.width, height=background_image.height)
    canvas.pack(fill='both', expand=True)
    canvas.create_image(0, 0, image=background_photo, anchor='nw')

    canvas.create_text(600, 50, text='Face recognition', font=('Arial', 40), fill='white')

    # Create buttons for collecting data and recognizing faces
    button1 = tk.Button(root, text='Input your ID and name', command=collect_and_train)
    button2 = tk.Button(root, text='Recognize Faces', command=recognize_faces)

    canvas.create_window(600, 150, window=button1)
    canvas.create_window(600, 200, window=button2)

    notice_label = tk.Label(root, text='', fg='red', bg='white')
    canvas.create_window(600, 250, window=notice_label)

    # Load the lists when the application starts
    load_data()

    root.mainloop()


# Run the GUI
if __name__ == '__main__':
    gui()