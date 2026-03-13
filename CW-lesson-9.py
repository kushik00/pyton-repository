import cv2

net = cv2.dnn.readNetFromCaffe('data/mobileNet/mobilenet_deploy.prototxt' , 'data/mobileNet/mobilenet.caffemodel')
classes = []

with open('data/mobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


image = cv2.imread('images/image.jpg')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))

net.setInput(blob)
preds = net.forward()

index = preds[0].argmax()

label = classes[index] if index < len(classes) else 'unknown'
conf = float(preds[0][index].item()) * 100

print(f'клас: {label} ')
print(f'ймовірність: {round(conf,2)}%')


text = label + ':' + str(round(conf,2)) + '%'
cv2.putText(image , text , (10,30) , cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255), 2)

cv2.imshow('result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
