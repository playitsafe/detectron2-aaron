import pandas as pd
import smtplib
import imghdr
from email.message import EmailMessage

img = cv2.imread('img15.png')
with open('img15.png', 'rb') as f:
  file_data = f.read()
  file_type = imghdr.what(f.name)
  file_name = f.name
# _ = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

outputs = predictor(img)
classes=outputs['instances'].pred_classes.cpu().numpy()
bbox=outputs['instances'].pred_boxes.tensor.cpu().numpy()
ind = np.where(classes==0)[0]
person=bbox[ind]
num= len(person)

#define a function which return the bottom center of every bbox
def mid_point(img,person,idx):
  #get the coordinates
  x1,y1,x2,y2 = person[idx]
  _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
  
  #compute bottom center of bbox
  x_mid = int((x1+x2)/2)
  y_mid = int(y2)
  mid   = (x_mid,y_mid)
  
  _ = cv2.circle(img, mid, 5, (0, 0, 255), -1)
  cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)
  
  return mid

  #call the function
midpoints = [mid_point(img,person,i) for i in range(len(person))]

# %%time
from scipy.spatial import distance
def compute_distance(midpoints,num):
  dist = np.zeros((num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist

dist= compute_distance(midpoints,num)
print(dist[0][1])

# %%time
def find_closest(dist,num,thresh):
  p1=[]
  p2=[]
  d=[]
  hasRisk = False
  for i in range(num):
    for j in range(i,num):
      if( (i!=j) & (dist[i][j]<=thresh)):
        p1.append(i)
        p2.append(j)
        d.append(dist[i][j])
        hasRisk = True
        # print(d)
  return p1,p2,d,hasRisk

# print(d)

thresh=100
p1,p2,d,hasRisk=find_closest(dist,num,thresh)
df = pd.DataFrame({"p1":p1,"p2":p2,"dist":d, "hasRisk": hasRisk})
df

def change_2_red(img,person,p1,p2):
  risky = np.unique(p1+p2)
  for i in risky:
    x1,y1,x2,y2 = person[i]
    _ = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)  
  return img

img = change_2_red(img,person,p1,p2)

plt.figure(figsize=(20,10))
plt.imshow(img)
new_img = img[..., ::-1]
cv2.imwrite('out/'+file_name,new_img)

def sendAlertEmail(image):
  msg = EmailMessage()
  msg['Subject'] = 'Violation of social distance detected! '
  msg['From'] = '137912334@qq.com'
  msg['To'] = 'aaron.ts@foxmail.com'
  msg.set_content('Violation of social distance detected! Plz check the image!')

  with open(image, 'rb') as f:
    out_file_data = f.read()
    out_file_type = imghdr.what(f.name)
    out_file_name = f.name

  msg.add_attachment(out_file_data, maintype='image', subtype=out_file_type, filename=out_file_name)

  with smtplib.SMTP_SSL('smtp.qq.com', 465) as smtp:
    smtp.login('137912334@qq.com', 'houllcezvqxzbgi')
    smtp.send_message(msg)

if hasRisk:
  print('Too close!!!!')

  # cv2.imwrite('out/'+file_name,img)

  sendAlertEmail('out/'+file_name)

else:
  print('It\'s safe!!!')