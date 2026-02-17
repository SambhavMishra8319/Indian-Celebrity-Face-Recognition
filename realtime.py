import cv2
import torch
import torchvision.transforms as transforms
from model import FaceCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FaceCNN(num_classes=3)
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = transform(face).unsqueeze(0).to(device)
        
        outputs = model(face)
        _, predicted = torch.max(outputs.data, 1)
        
        label = str(predicted.item())
        
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h),
                      (255,0,0), 2)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
