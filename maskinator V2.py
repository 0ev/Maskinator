import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
import uuid
from upload import upload
import threading
import screeninfo
from colors import closest_color
from speaker import Speaker


window_name = "frame"

voice = True

screen = screeninfo.get_monitors()[0]

cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def sampling(x1, y1, x2, y2):
    # x1, y1, x2, y2 = R1[0], R1[1], R2[0], R2[1]
    fx1, fy1, fx2, fy2 = (int((3*x1 + x2)/4),int((3*y2 - y1)/2), int((x1 + 3*x2)/4), int((2*y2 - y1)/1))
    if fx1 > 640:
        fx1 = 640
    if fx1 < 0:
        fx1 = 0
    if fx2 > 640:
        fx2 = 640
    if fx2 < 0:
        fx2 = 0
    if fy1 > 360:
        fy1 = 360
    if fy1 < 0:
        fy1 = 0
    if fy2 > 360:
        fy2 = 360
    if fy2 < 0:
        fy2 = 0
    if fx2<=fx1:
        return None
    if fy2<=fy1:
        return None
    return (fx1, fy1, fx2, fy2)

def is_sim(R1,R2):
    '''
    checks if two faces are the same
    '''
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    else:
        return True

def main(model_test,net,cap,crop_img_base):
    '''
    main function
    '''

    color_speaker = Speaker()
    person_dic = {} # dictionary that stores every person
    color_dic = {"aqua": "하늘색",
    "black": "검은색",
    "blue": "파란색",
    "fuchsia": "보라색",
    "green": "초록색",
    "gray": "하얀색",
    "lime": "연두색",
    "maroon": "갈색",
    "navy": "남색",
    "olive": "카키색",
    "purple": "보라색",
    "red": "빨간색",
    "silver": "하얀색",
    "teal": "초록색",
    "white": "하얀색",
    "yellow": "노란색",
    "aliceblue":"하얀색",
    "antiquewhite":"하얀색",
    "aqua":"하늘색",
    "aquamarine":"연두색",
    "azure":"하얀색",
    "beige":"하얀색",
    "bisque":"하얀색",
    "black":"검은색",
    "blanchedalmond":"하얀색",
    "blue":"파란색",
    "blueviolet":"보라색",
    "brown":"빨간색",
    "burlywood":"베이지색",
    "cadetblue":"하늘색",
    "chartreuse":"초록색",
    "chocolate":"주황색",
    "coral":"주황색",
    "cornflowerblue":"파랑색",
    "cornsilk":"하양색",
    "crimson":"빨간색",
    "cyan":"하늘색",
    "darkblue":"남색",
    "darkcyan":"하늘색",
    "darkgoldenrod":"노란색",
    "darkgray":"하얀색",
    "darkgrey":"하얀색",
    "darkgreen":"초록색",
    "darkkhaki":"노란색",
    "darkmagenta":"보라색",
    "darkolivegreen":"카키색",
    "darkorange":"주황색",
    "darkorchid":"보라색",
    "darkred":"빨산색",
    "darksalmon":"주황색",
    "darkseagreen":"연두색",
    "darkslateblue":"보라색",
    "darkslategray":"청록색",
    "darkslategrey":"청록색",
    "darkturquoise":"하늘색",
    "darkviolet":"보라색",
    "deeppink":"핑크색",
    "deepskyblue":"하늘색",
    "dimgray":"회색",
    "dimgrey":"회색",
    "dodgerblue":"파란색",
    "firebrick":"빨간색",
    "floralwhite":"하얀색",
    "forestgreen":"초록색",
    "fuchsia":"핑크색",
    "gainsboro":"하얀색",
    "ghostwhite":"하얀색",
    "gold":"노란색",
    "goldenrod":"노란색",
    "gray":"하얀색",
    "grey":"하얀색",
    "green":"초록색",
    "greenyellow":"연두색",
    "honeydew":"하얀색",
    "hotpink":"핑크색",
    "indianred":"분홍색",
    "indigo":"보라색",
    "ivory":"흰색",
    "khaki":"노랑색",
    "lavender":"하얀색",
    "lavenderblush":"하얀색",
    "lawngreen":"연두색",
    "lemonchiffon":"하얀색",
    "lightblue":"하늘색",
    "lightcoral":"분홍색",
    "lightcyan":"하얀색",
    "lightgoldenrodyellow":"하얀색",
    "lightgray":"하얀색",
    "lightgrey":"하얀색",
    "lightgreen":"연두색",
    "lightpink":"핑크색",
    "lightsalmon":"주황색",
    "lightseagreen":"청록색",
    "lightskyblue":"하늘색",
    "lightslategray":"하얀색",
    "lightslategrey":"하얀색",
    "lightsteelblue":"하얀색",
    "lightyellow":"하얀색",
    "lime":"연두색",
    "limegreen":"연두색",
    "linen":"하얀색",
    "magenta":"보라색",
    "maroon":"빨간색",
    "mediumaquamarine":"몰라",
    "mediumblue":"파란색",
    "mediumorchid":"핑크색"
    }

    img_color_dic = {"aqua": "light blue",
    "black": "black",
    "blue": "blue",
    "fuchsia": "purple",
    "green": "green",
    "gray": "grey",
    "lime": "light grey",
    "maroon": "brown",
    "navy": "dark blue",
    "olive": "khaki",
    "purple": "purple",
    "red": "red",
    "silver": "white",
    "teal": "green",
    "white": "white",
    "yellow": "yellow",
    "aliceblue":"White",
    "antiquewhite":"White",
    "aqua":"skyblue",
    "aquamarine":"Light green",
    "azure":"White",
    "beige":"White",
    "bisque":"White",
    "black":"a dark color",
    "blanchedalmond":"White",
    "blue":"Blue",
    "blueviolet":"Purple",
    "brown":"red",
    "burlywood":"beige color",
    "cadetblue":"skyblue",
    "chartreuse":"Green",
    "chocolate":"Orange",
    "coral":"Orange",
    "cornflowerblue":"Blue",
    "cornsilk":"White color",
    "crimson":"red",
    "cyan":"skyblue",
    "darkblue":"Indigo",
    "darkcyan":"skyblue",
    "darkgoldenrod":"Yellow",
    "darkgray":"White",
    "darkgrey":"White",
    "darkgreen":"Green",
    "darkkhaki":"Yellow",
    "darkmagenta":"Purple",
    "darkolivegreen":"khaki color",
    "darkorange":"Orange",
    "darkorchid":"Purple",
    "darkred":"Red",
    "darksalmon":"Orange",
    "darkseagreen":"Light green",
    "darkslateblue":"Purple",
    "darkslategray":"Turquoise",
    "darkslategrey":"Turquoise",
    "darkturquoise":"skyblue",
    "darkviolet":"Purple",
    "deeppink":"Pink",
    "deepskyblue":"skyblue",
    "dimgray":"Gray",
    "dimgrey":"Gray",
    "dodgerblue":"Blue",
    "firebrick":"red",
    "floralwhite":"White",
    "forestgreen":"Green",
    "fuchsia":"Pink",
    "gainsboro":"White",
    "ghostwhite":"White",
    "gold":"Yellow",
    "goldenrod":"Yellow",
    "gray":"White",
    "grey":"White",
    "green":"Green",
    "greenyellow":"Light green",
    "honeydew":"White",
    "hotpink":"Pink",
    "indianred":"Pink",
    "indigo":"Purple",
    "ivory":"White",
    "khaki":"Yellow",
    "lavender":"White",
    "lavenderblush":"White",
    "lawngreen":"Light green",
    "lemonchiffon":"White",
    "lightblue":"skyblue",
    "lightcoral":"Pink",
    "lightcyan":"White",
    "lightgoldenrodyellow":"White",
    "lightgray":"White",
    "lightgrey":"White",
    "lightgreen":"Light green",
    "lightpink":"Pink",
    "lightsalmon":"Orange",
    "lightseagreen":"Turquoise",
    "lightskyblue":"skyblue",
    "lightslategray":"White",
    "lightslategrey":"White",
    "lightsteelblue":"White",
    "lightyellow":"White",
    "lime":"Light green",
    "limegreen":"Light green",
    "linen":"White",
    "magenta":"Purple",
    "maroon":"red",
    "mediumaquamarine":"Don't know",
    "mediumblue":"Blue",
    "mediumorchid":"Pink",
    }

    # mainloop

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

        # iterate through detections and preprocess them

        for i in range(0, detections.shape[2]):

            try:

                confidence = detections[0, 0, i, 2]

                if confidence < 0.3:
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                
                person = None

                initial = [box,10,10,None]
                
                if not person_dic:
                    person = uuid.uuid4() # give the person a unique identifier
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
                        person = uuid.uuid4() # give the person a unique identifier
                        print(f"new person! : {person}")
                        person_dic[person] = initial


                startX, startY, endX, endY = box.astype("int")

                sample = sampling(startX, startY, endX, endY)

                if voice:

                    if sample != None:

                        x1, y1, x2, y2 = sample
                        sample = frame[y1:y2, x1:x2].copy()
                        color_clothing = unique_count_app(sample)
                        color_name = closest_color(color_clothing)
                        color_name_for_text = img_color_dic[color_name]
                        cv2.putText(frame, color_name_for_text, (x1, y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,0), 2)
                        person_dic[person][3] = color_dic[color_name]

                
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
                raise err
                print(err)
                pass
        
        # detect faces and draw rectangles and put text

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
                    if voice:
                        color_speaker.speak(person_dic[person][3])

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
            print(f"uploading {x[0]} to board")
                
        cv2.imshow(window_name, frame)
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
        except AttributeError:
            break
        except Exception as err:
            raise err
            print(err)
            pass
