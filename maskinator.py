import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
import uuid
from upload import upload
import threading

def is_sim(R1,R2):
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    else:
        return True

def main(model_test,net,cap,crop_img_base):


    person_dic = {}

    while True:

        _, frame = cap.read()
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (640,360), (104.0,177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        crop_img = crop_img_base

        list_faces = []
        list_boxes = []
        list_send = []
        real_faces = []
        face_counter = 0

        for i in range(0, detections.shape[2]):

            try:

                confidence = detections[0, 0, i, 2]

                if confidence < 0.5:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # box = np.array([box[0]-10, box[1]-10, box[2]+10, box[3]+10])
                
                person = None

                initial = [box,10,10]
                
                if not person_dic:
                    person = uuid.uuid4()
                    person_dic[person] = initial

                else:
                    check = 0
                    for x in person_dic:
                        if is_sim(person_dic[x][0],box):
                            person = x
                            person_dic[x][0] = box
                            person_dic[x][1] = 10
                            check += 1

                    if check == 0:
                        person = uuid.uuid4()
                        print(f"new person! : {person}")
                        person_dic[person] = initial


                startX, startY, endX, endY = box.astype("int")

                if abs(startY-endY) > 180:
                    textsize = cv2.getTextSize("TOO CLOSE MOVE AWAY!", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    textX = int(320 - (textsize[0] / 2))
                    textY = int(180 + (textsize[1] / 2))
                    cv2.putText(frame,"TOO CLOSE MOVE AWAY!",(textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    continue

                if max(startX,endX)>=640 or 0>=min(startX,endX):
                    continue
                if max(startY,endY)>=360 or 0>=min(startY,endY):
                    continue

                crop_img = frame[startY:endY, startX:endX].copy()

                img_array = crop_img
                img_array = cv2.resize(img_array, dsize=(128,128), interpolation=cv2.INTER_LINEAR)
                img_array = img_array.astype('float')            
                img_array /= 255.0

                list_faces.append(img_array)
                real_faces.append(crop_img)

                list_boxes.append((startX, startY, endX, endY, person))

                face_counter += 1

            except Exception as err:
                print(err)
                pass
        
        #draw rectangles and put text

        if face_counter:

            list_faces = np.array(list_faces)
            list_faces = list_faces.reshape(face_counter,128,128,3)
            result = model_test.predict(list_faces)
            for i,x in enumerate(result):

                if x[0] > 0.9:
                    is_mask = True
                else:
                    is_mask = False

                if is_mask:
                    color = (255,0,0)
                else:
                    color = (0,0,255)

                
                startX, startY, endX, endY, person = list_boxes[i]

                y = startY - 10 if startY - 10 > 10 else startY + 10

                text = f"{str(person)[:3]}"

                if (not is_mask) and person_dic[person][2] > 0:
                    person_dic[person][2] -= 1
                    text += "  "+str(person_dic[person][2])


                elif person_dic[person][2] == 0:
                    list_send.append([person, real_faces[i]])
                    person_dic[person][2] = -1

                elif person_dic[person][2] == -1:
                    text += "  dead"

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # upload image to website

        if person_dic:
            k = []
            for x in person_dic:
                if person_dic[x][1] == 0:
                    k.append(x)

            for x in k:
                del person_dic[x]

            for x in person_dic:
                person_dic[x][1] -= 1

        for i,x in enumerate(list_send):
            threading.Thread(target=upload, args=(x[0],x[1])).start()
            print("uploading")
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    with open("models/mask_detection/model.json", "r") as fp:
        model_test = model_from_json(fp.read())

    model_test.load_weights("models/mask_detection/final.h5")

    net = cv2.dnn.readNetFromCaffe('models/face_detection/deploy.prototxt.txt', 'models/face_detection/res10_300x300_ssd_iter_140000.caffemodel')

    cap = cv2.VideoCapture(0) 
    cap.set(3, 640)
    cap.set(4, 360)

    crop_img_base = cv2.imread("download.png")

    while True:
        try:
            main(model_test,net,cap,crop_img_base)
        except KeyboardInterrupt:
            break
        except Exception as err:
            print(err)
            pass
