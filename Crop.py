
## 终版；可指定开始文件夹
import os
import json
import cv2

def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def extract_face_and_body(image, face_bbox, body_bbox):
    # 检查边界框是否在图像内
    h, w = image.shape[:2]
    face_bbox = [max(0, min(w-1, int(coord))) for coord in face_bbox[:2]] + [max(0, min(h-1, int(coord))) for coord in face_bbox[2:]]
    body_bbox = [max(0, min(w-1, int(coord))) for coord in body_bbox[:2]] + [max(0, min(h-1, int(coord))) for coord in body_bbox[2:]]

    face = image[face_bbox[1]:face_bbox[1]+face_bbox[3], face_bbox[0]:face_bbox[0]+face_bbox[2]]
    body = image[body_bbox[1]:body_bbox[1]+body_bbox[3], body_bbox[0]:body_bbox[0]+body_bbox[2]]
    return face, body

def process_data(data_dir, annotation_dir, start_from=0):
    for subdir in sorted(os.listdir(data_dir)):
        if is_number(subdir):
            subdir_num = int(subdir)
            if subdir_num < start_from:
                continue

            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                for i in range(45):
                    image_path = os.path.join(subdir_path, f'incarframes/{i}.jpg')
                    annotation_path = os.path.join(annotation_dir, f'{subdir}.json')

                    if os.path.exists(image_path) and os.path.exists(annotation_path):
                        print(f"正在处理 {image_path} 和 {annotation_path}")

                        image = cv2.imread(image_path)
                        annotations = load_annotations(annotation_path)

                        for pose in annotations['pose_list']:
                            img_name = pose['imgname']
                            if img_name == f"{i}.jpg":
                                face_bbox = pose['result'][0]['face_bbox']
                                body_bbox = pose['result'][0]['bbox']

                                face, body = extract_face_and_body(image, face_bbox, body_bbox)

                                face_output_path = os.path.join(subdir_path, "face", f'{img_name.replace(".jpg", "_face.jpg")}')
                                body_output_path = os.path.join(subdir_path, "body", f'{img_name.replace(".jpg", "_body.jpg")}')

                                os.makedirs(os.path.dirname(face_output_path), exist_ok=True)
                                os.makedirs(os.path.dirname(body_output_path), exist_ok=True)

                                # 检查图像是否为空
                                if face.size > 0 and not os.path.exists(face_output_path):
                                    cv2.imwrite(face_output_path, face)
                                if body.size > 0 and not os.path.exists(body_output_path):
                                    cv2.imwrite(body_output_path, body)
                                if face.size > 0 or body.size > 0:
                                    print(f"脸部和身体图片已保存: {face_output_path}, {body_output_path}")
                                else:
                                    print(f"图像为空，跳过: {subdir}/{img_name}")

data_directory = "./"
annotation_directory = os.path.join(data_directory, "annotation")

process_data(data_directory, annotation_directory, start_from=2441)

