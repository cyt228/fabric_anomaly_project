import cv2, csv
import numpy as np
import os

def random_region(img,
                  shape_types = ("circle","vert_strip","hori_strip"), # rect
                  area_ratio_range = (0.1, 0.2), # 區域面積佔整張圖比例(最小, 最大)
                  rng: np.random.Generator = None):
    
    H, W = img.shape[:2]
    if rng is None:
        rng = np.random.default_rng()  # 每次不同

    # 決定面積
    a_min, a_max = area_ratio_range
    target_area = float(rng.uniform(a_min, a_max)) * H * W
    #print(target_area)

    # 決定形狀
    shape = rng.choice(shape_types)
    #print(shape)

    mask = np.zeros((H, W), dtype = np.uint8)

    def place_rect(w, h):
        w = max(1, min(int(w), W))
        h = max(1, min(int(h), H))
        x = int(rng.integers(0, max(1, W - w)))
        y = int(rng.integers(0, max(1, H - h)))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return x, y, w, h

    if shape == "vert_strip":
        # 直條
        aspect = float(rng.uniform(5, 80)) # 分配長寬比
        w = max(1, int(np.sqrt(target_area / aspect))) # aspect越大，w越小
        h = max(2, int(w * aspect))
        x, y, w, h = place_rect(w, h)
        #print(x," ",y," ",w," ",h)
    
    elif shape == "hori_strip":
        # 橫條
        aspect = float(rng.uniform(5, 80))
        h = max(1, int(np.sqrt(target_area / aspect)))
        w = max(2, int(h * aspect))
        x, y, w, h = place_rect(w, h)
    
    elif shape == "rect":
        # 一般矩形
        ar = float(rng.uniform(0.8, 5))  # 長寬比
        w = np.sqrt(target_area * ar)
        h = target_area / w
        x, y, w, h = place_rect(w, h)
    
    elif shape == 'circle':
        # 圓形
        r = int(max(1, np.sqrt(target_area / np.pi))) # 半徑
        cx = int(rng.integers(r, W - r))
        cy = int(rng.integers(r, H - r))
        cv2.circle(mask, (cx, cy), r, 255, -1)
    
    #output = cv2.add(img, mask)  
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0) 
    return mask


def add_salt_pepper_in_mask(img,
                            mask,
                            density = 0.3, #雜訊密度
                            salt_vs_pepper = 0.5, # salt比例
                            rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    noisy = img.copy()
    size = noisy.size
    num_salt = np.ceil(density * size * salt_vs_pepper).astype('int')
    num_pepper = np.ceil(density * size * (1 - salt_vs_pepper)).astype('int')
    row, column = noisy.shape[:2]

     # 隨機的座標點
    x = np.random.randint(0, column - 1, num_pepper)
    y = np.random.randint(0, row - 1, num_pepper)
    noisy[y, x] = 0   # 撒上胡椒

    # 隨機的座標點
    x = np.random.randint(0, column - 1, num_salt)
    y = np.random.randint(0, row - 1, num_salt)
    noisy[y, x] = 255 # 撒上鹽

    #cv2.imshow('img with noisy', noisy)
    #cv2.waitKey(0)

    wight_img = cv2.bitwise_and(noisy, noisy, mask=mask)              # 白區：noisy
    black_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask)) # 黑區：原圖
    noisy_img = cv2.add(wight_img, black_img)
    #cv2.imshow('img add noisy', noisy_img)
    #cv2.waitKey(0)

    return noisy_img

def make_defect(img):
    mask = random_region(img)

    rng = np.random.default_rng()
    select_defect = rng.integers(0, 2)

    if select_defect == 0:
        #print("noisy")
        result_img = add_salt_pepper_in_mask(img, mask)
    elif select_defect == 1:
        #print("mask")
        mask = 255 - mask #黑白轉換
        result_img = cv2.bitwise_and(img, img, mask=mask)
        #cv2.imshow('img_mask', result_img)
        #cv2.waitKey(0)
    #img_noisy_mask = cv2.bitwise_and(img, img, mask=noisy_mask)
    #cv2.imshow('img noisy', img_noisy_mask)
    #cv2.waitKey(0)

    return result_img

in_path = 'dataset/SP3/train/defect-free/'
png_files = []
for filename in os.listdir(in_path):
    #file_path = os.path.join(in_path,filename)
    png_files.append(filename)
print(png_files)

#out_path = 'dataset/SP3/train/defect/'
''''''
out_path = 'dataset/SP3/train/defect/'
csv_path = 'dataset/SP3/train/defect/no_defects.csv'
OVERWRITE = True
os.makedirs(out_path, exist_ok = True)
''''''
mode = "w" if OVERWRITE or (not os.path.exists(csv_path)) else "a"
fcsv = open(csv_path, mode, newline="", encoding="utf-8")
writer = csv.writer(fcsv)
if mode == "w":
    writer.writerow(["filename"])   # 表頭（只記錄檔名）

for pic in png_files:
    img = cv2.imread(in_path + pic, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read image")

    rng = np.random.default_rng()
    num_of_defect = rng.integers(1, 5)
    if num_of_defect == 4:
        cv2.imwrite(out_path + pic, img)
        writer.writerow([pic])
        print("How many defect are made:  0")
    else:
        print("How many defect are made: ",num_of_defect)

        for i in range(num_of_defect):
            img = make_defect(img)
            #cv2.imshow('result_img', img)
            #cv2.waitKey(0)

        cv2.imwrite(out_path + pic, img)
    