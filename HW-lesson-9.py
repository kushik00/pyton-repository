import cv2

net = cv2.dnn.readNetFromCaffe('data/mobileNet/mobilenet_deploy.prototxt', 'data/mobileNet/mobilenet.caffemodel')

classes = []
with open('data/mobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split(' ', 1)
        classes.append(parts[1] if len(parts) > 1 else parts[0])

image_paths = ['photos/1.jpg', 'photos/apples.jpg', 'photos/green.jpg']

for i, path in enumerate(image_paths):
    image = cv2.imread(path)
    if image is None:
        print(f"Помилка: не вдалося завантажити {path}")
        continue

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))

    net.setInput(blob)
    preds = net.forward()

    index = preds[0].argmax()
    label = classes[index] if index < len(classes) else 'unknown'
    conf = float(preds.flatten()[index]) * 100

    print(f'Файл: {path} | Клас: {label} | Ймовірність: {round(conf, 2)}%')

    text = f"{label}: {round(conf, 2)}%"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    cv2.imshow(f'Result {i + 1}', image)

print(" ")
cv2.waitKey(0)
cv2.destroyAllWindows()
