from flask import Flask, jsonify, request
from flask_cors import CORS
from http import HTTPStatus
from transformers import VisionEncoderDecoderModel,TrOCRProcessor
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2 import service_account
import os
import cv2
import re
import pandas as pd
import numpy as np
import base64
import random
import json
import logging

load_dotenv()
app = Flask(__name__)
CORS(app)
app.config['MODEL_OBJECT_DETECTION']='./model/detect_plat.pt'
app.config['MODEL_OCR'] = os.environ.get('MODEL_OCR')
app.config['UPLOAD_IMAGES_OBJECT_DETECTION'] = './image/object-detect/images/'
app.config['UPLOAD_IMAGES_OCR'] = './image/OCR/images/'

bucket_name = os.environ.get('BUCKET_NAME_AP','bucket-automation-parking')
credentials_json = os.environ.get('CREDENTIALS')

credentials_dict = json.loads(credentials_json)
credentials = service_account.Credentials.from_service_account_info(credentials_dict)
client = storage.Client(credentials=credentials)

bucket = storage.Bucket(client,bucket_name)

model_detect = YOLO(app.config['MODEL_OBJECT_DETECTION'])
detect_names = model_detect.names

processor_ocr = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
model_ocr = VisionEncoderDecoderModel.from_pretrained(app.config['MODEL_OCR'])

# Peta Wilayah dan Daerah
sumatera_map = {
    "BL": "Aceh", "BB": "Sumatera Utara bagian barat", "BK": "Sumatera Utara bagian timur",
    "BA": "Sumatera Barat", "BM": "Riau", "BH": "Jambi", "BD": "Bengkulu", "BP": "Kepulauan Riau",
    "BG": "Sumatera Selatan", "BE": "Lampung"
}
banten_map = { "A": "Banten, Cilegon, Serang, Pandeglang, Lebak, Tangerang" }
jakarta_map = { "B": "Jakarta, Depok, Bekasi" }
jawa_barat_map = {
    "D": "Bandung", "E": "Cirebon, Majalengka, Indramayu, Kuningan", "F": "Bogor, Cianjur, Sukabumi",
    "T": "Purwakarta, Karawang, Subang", "Z": "Garut, Sumedang, Tasikmalaya, Pangandaran, Ciamis, Banjar"
}
jawa_tengah_map = {
    "G": "Pekalongan, Pemalang, Batang, Tegal, Brebes", "H": "Semarang, Kendal, Salatiga, Demak",
    "K": "Pati, Jepara, Kudus, Blora, Rembang", "R": "Banyumas, Purbalingga, Cilacap, Banjarnegara",
    "AA": "Magelang, Purworejo, Temanggung, Kebumen, Wonosobo", "AD": "Surakarta, Sukoharjo, Boyolali, Klaten"
}
jogja_map = { "AB": "Yogyakarta, Bantul, Gunung Kidul, Sleman, Kulon Progo" }
jawa_timur_map = {
    "L": "Surabaya", "M": "Madura", "N": "Malang, Pasuruan, Probolinggo, Lumajang",
    "P": "Bondowoso, Jember, Situbondo, Banyuwangi", "S": "Bojonegoro, Tuban, Mojokerto, Lamongan",
    "W": "Gresik, Sidoarjo", "AE": "Madiun, Ngawi, Ponorogo", "AG": "Kediri, Blitar, Tulungagung"
}
bali_nusa_map = {
    "DK": "Bali", "DR": "Pulau Lombok, Mataram", "EA": "Pulau Sumbawa", "DH": "Pulau Timor, Kupang",
    "EB": "Pulau Flores", "ED": "Pulau Sumba"
}
kalimantan_map = {
    "KB": "Singkawang, Pontianak", "DA": "Banjarmasin", "KH": "Palangkaraya, Kotawaringin, Barito",
    "KT": "Balikpapan, Samarinda, Bontang", "KU": "Kalimantan Utara"
}
sulawesi_map = {
    "DB": "Manado", "DL": "Sitaro, Talaud", "DM": "Gorontalo", "DN": "Palu, Poso",
    "DT": "Kendari, Konawe", "DD": "Makassar", "DC": "Majene"
}
maluku_papua_map = {
    "DE": "Maluku", "DG": "Ternate, Tidore", "PA": "Jayapura, Merauke", "PB": "Papua Barat"
}

maps = {
    "Sumatera": sumatera_map,
    "Banten": banten_map,
    "Jakarta": jakarta_map,
    "Jawa Barat": jawa_barat_map,
    "Jawa Tengah": jawa_tengah_map,
    "Yogyakarta": jogja_map,
    "Jawa Timur": jawa_timur_map,
    "Bali dan Nusa Tenggara": bali_nusa_map,
    "Kalimantan": kalimantan_map,
    "Sulawesi": sulawesi_map,
    "Maluku dan Papua": maluku_papua_map
}


def find_area(plate_area):
    for region, city in maps.items():
        if plate_area in city:
            return region, city[plate_area]
    return None,None

def regex_plat(nomor_plat):
    match = re.match(r"([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})", nomor_plat)
    if match:
        return match.group(1), match.group(2), match.group(3)
    else:
        return None,None,None

def vehicle_classification(angka):
    angka = int(angka)
    if 1 <= angka <= 1999:
        return "Mobil Penumpang"
    elif 2000 <= angka <= 6999:
        return "Sepeda Motor"
    elif 7000 <= angka <= 7999:
        return "Bus"
    elif 8000 <= angka <= 9999:
        return "Mobil Pengangkut"
    else:
        return "Tidak diketahui"


def crop(image_path):
    
    original_filename, ext = os.path.splitext(os.path.basename(image_path))

    im0 = cv2.imread(image_path)
    if im0 is None:
        raise ValueError(f"Error reading image file {image_path}")

    results = model_detect.predict(im0, show=False)
    logging.info(f"Prediksi OD: {results}")
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=detect_names)

    idx = 0
    cropped_files = []
    print(boxes)
    if boxes != []:
        for box, cls in zip(boxes, clss):
            idx += 1
            class_name = detect_names[int(cls)] 
            annotator.box_label(box, color=colors(int(cls), True), label=class_name)
            crop_obj = im0[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            crop_filename = os.path.join(app.config['UPLOAD_IMAGES_OCR'], f"{original_filename}{idx}{ext}")
            cv2.imwrite(crop_filename, crop_obj)
            cropped_files.append(crop_filename)
        return crop_obj,crop_filename
    else:
        return None,None


def ocr(image):

    pixel_values = processor_ocr(image, return_tensors='pt').pixel_values
    generated_ids = model_ocr.generate(pixel_values)
    generated_text = processor_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logging.info(f"prediksi OCR : {generated_text}")
    return generated_text



@app.route("/")
def index():
    return jsonify({
        "status" : {
            "code" : HTTPStatus.OK,
            "message" : "nyambung cuy santui",
        },
        "data" : None
    }),HTTPStatus.OK

@app.route("/prediction",methods=["POST"]) 
def predict():
    if request.method == 'POST':
        reqImage =  request.get_json()
        image_data = reqImage.get('image')
        logging.info(f"data masuk : {reqImage}")
        if image_data:
            if "data:image" in image_data:
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            np_img = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            file_name= str(random.randint(10000,99999))+"image_object-detect.png"
            file_path = app.config['UPLOAD_IMAGES_OBJECT_DETECTION']+file_name
            cv2.imwrite(file_path, img)
            
            file_path_OD = f"object-detect/{file_name}"
            blob_OD = bucket.blob(file_path_OD)
            blob_OD.upload_from_filename(file_path)
            
            crop_image,crop_filename = crop(file_path)
            if crop_image is not None and crop_filename is not None :
                plate = ocr(crop_image)
                area,type,temp = regex_plat(plate)
                region,city_province = find_area(area)
                vehicle_type = vehicle_classification(type)
                
                ocr_filename =os.path.basename(crop_filename)
                file_path_OCR = "OCR/"+ocr_filename   
                
                
                
                blob_OCR = bucket.blob(file_path_OCR)
                blob_OCR.upload_from_filename(crop_filename)
                
                os.remove(file_path)
                os.remove(crop_filename)
                
                result = {
                    'platNomor' : plate,
                    'wilayah' : region,
                    'kota_provinsi' : city_province,
                    'jenis_kendaraan' : vehicle_type,
                    'image_link' : 'https://storage.googleapis.com/'+'bucket-automation-parking/OCR/'+ ocr_filename
                }
                return jsonify({
                                'status': {
                                    'code': HTTPStatus.OK,
                                    'message': 'Success predicting',
                                },
                                'data': result
                            }),HTTPStatus.OK,
            else :
                os.remove(file_path)
                result = {
                        'platNomor' : None,
                        'wilayah' : None,
                        'kota_provinsi' : None,
                        'jenis_kendaraan' : None,
                        'image_link' : 'https://storage.googleapis.com/'+'bucket-automation-parking/object-detect/'+ file_name
                        }
                return jsonify({
                            "status" : {
                                "code" : HTTPStatus.OK,
                                "message" : "NO detect",
                            },
                            "data" : result
                        }),HTTPStatus.OK
        else:
            return jsonify({
                            'status': {
                                'code': HTTPStatus.BAD_REQUEST,
                                'message': 'Invalid file format',
                            }
                            }),HTTPStatus.BAD_REQUEST,
                
    else:
        return jsonify({
                        'status': {
                            'code': HTTPStatus.METHOD_NOT_ALLOWED,
                            'message': 'Methode not allowed',
                        }
                        }),HTTPStatus.METHOD_NOT_ALLOWED,

@app.route("/datarecap",methods=["POST"]) 
def DataRecap():
    data = request.get_json()
    name_file = data.get('filename')+'.xlsx'
    df = pd.DataFrame(data.get('data'))
    
    file_path = f"excel/{name_file}"
    excel_file_path = f'Excel-folder/{name_file}'
    df.to_excel(excel_file_path, index=False)
    blob_excel = bucket.blob(file_path)
    blob_excel.upload_from_filename(excel_file_path)
    os.remove(excel_file_path)
    result = {
            'file_link' : 'https://storage.googleapis.com/'+'bucket-automation-parking/'+ file_path
                }
    return jsonify({
                    'status': {
                        'code': HTTPStatus.OK,
                        'message': 'Success predicting',
                    },
                    'data': result
                    }),HTTPStatus.OK,
    
if __name__ == '__main__': 
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0', port=port)
