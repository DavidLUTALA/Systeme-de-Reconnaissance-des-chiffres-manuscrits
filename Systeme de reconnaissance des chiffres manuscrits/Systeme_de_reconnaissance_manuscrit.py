import tkinter as tk
import numpy as np
from PIL import ImageGrab, Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Charger et préparer le dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
# Normalisation des pixels (entre 0 et 1)
x_test = x_test / 255.0

# Ajouter une dimension pour correspondre au format attendu (canal d'entrée)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convertir les étiquettes en encodage catégoriel
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Création d'un générateur de données avec augmentation
datagen = ImageDataGenerator(
    rotation_range=10,          # Rotation aléatoire de l'image dans un intervalle de 10°
    width_shift_range=0.1,      # Déplacement horizontal de 10%
    height_shift_range=0.1,     # Déplacement vertical de 10%
    zoom_range=0.1,             # Zoom aléatoire
    shear_range=0.1,            # Cisaillement aléatoire
    horizontal_flip=True,       # Retourner les images horizontalement
    fill_mode='nearest'         # Remplir les pixels vides après transformation
)

# Ajuster l'augmentation de données sur l'ensemble d'entraînement
datagen.fit(x_train)

# Construire le modèle CNN
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle en utilisant le générateur de données
model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=30, validation_data=(x_test, y_test), verbose=2)

# Sauvegarder le modèle après l'entraînement
model.save("digit_recognition_model_augmented_all_data.h5")


# Classe pour l'interface utilisateur
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Reconnaissance de Chiffres Manuscrits")

        # Créer un canevas pour dessiner
        self.canvas = tk.Canvas(self.root, width=300, height=300, bg='white')
        self.canvas.grid(row=0, column=0, pady=2, sticky="W")

        # Ajouter des boutons
        self.button_clear = tk.Button(self.root, text="Effacer", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.button_predict = tk.Button(self.root, text="Prédire", command=self.predict_digit)
        self.button_predict.grid(row=2, column=0, pady=2)

        # Événements pour dessiner
        self.canvas.bind("<B1-Motion>", self.draw)

    def clear_canvas(self):
        self.canvas.delete("all")

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # Taille du point
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

    def predict_digit(self):
        # Capturer le dessin du canevas
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))

        # Prétraiter l'image
        img = img.convert('L')  # Convertir en niveaux de gris
        img = img.resize((28, 28))  # Redimensionner à 28x28
        img_array = np.array(img)
        
        # Inverser les couleurs (noir sur blanc) pour correspondre à MNIST
        img_array = 255 - img_array  # Inverser les couleurs
        img_array = img_array / 255.0  # Normaliser les pixels entre 0 et 1
        
        # Aplatir l'image pour correspondre au format attendu par le modèle (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Faire une prédiction
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)

        # Afficher le résultat
        result = f"Chiffre prédit : {digit}"
        self.show_prediction(result, img_array)

    def show_prediction(self, result, img_array):
        prediction_label = tk.Label(self.root, text=result, font=("Helvetica", 16))
        prediction_label.grid(row=3, column=0, pady=2)
        
        # Afficher l'image prédit
        plt.imshow(img_array.reshape(28, 28), cmap='gray')
        plt.title(f"Prédiction: {result}")
        plt.show()


# Charger le modèle pré-entraîné (si déjà sauvegardé)
model = load_model("digit_recognition_model_augmented_all_data.h5")

# Lancer l'interface Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root, model)
    root.mainloop()
