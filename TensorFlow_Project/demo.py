import base64
from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager

from scipy import misc

from wide_resnet import WideResNet
import facenet
from keras.utils.data_utils import get_file
import tensorflow as tf
from align import detect_face

from PIL import Image
import flask
import io
import os
import pickle

app = flask.Flask(__name__)

model = None
detector = None
margin = None
celeb_model = None
classifier = None


@app.route("/predict", methods=["POST"])
def predict():
    # view로부터 반환될 데이터 딕셔너리를 초기화합니다.
    data = {"success": False}

    # 이미지가 엔트포인트에 올바르게 업로드 되었는쥐 확인하세요
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # PIL 형식으로 이미지를 읽어옵니다.
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image.save("input.jpg")
            opencv_img = np.array(image)
            image = opencv_img[:, :, ::-1].copy()
            # image = np.array(image)
            celeb_string = process_celebrity(image)
            data["celebrity"] = celeb_string
            h, w, _ = image.shape
            r = 640 / max(w, h)
            cv2.resize(image, (int(w * r), int(h * r)))

            result_image = process_age_gender(image)
            result_image = result_image[:, :, ::-1].copy()
            result_image = Image.fromarray(result_image)
            result_image.save("result.jpg")
            buffered = io.BytesIO()
            result_image.save(buffered, format="JPEG")
            img_bytes = base64.b64encode(buffered.getvalue())
            img_str = img_bytes.decode('ascii')
            data["image"] = img_str
            buffered.close()

            # 요청이 성공했음을 나타냅니다.
            data["success"] = True

    # JSON 형식으로 데이터 딕셔너리를 반환합니다.
    return flask.jsonify(data)


def process_age_gender(img):
    img_size = 64
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    "M" if predicted_genders[i][0] < 0.5 else "F")
            draw_label(img, (d.left(), d.top()), label)

        return img
    else:
        return img


def process_celebrity(img):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_size = np.asarray(img.shape)[0:2]
    img_list = []
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    count = len(bounding_boxes)
    for i in range(count):
        det = np.squeeze(bounding_boxes[i, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        # prewhitened = np.array(prewhitened).reshape(160, 160, 3)
        img_list.append(prewhitened)
    prewhitened = np.stack(img_list)

    with tf.Session() as sess:
        facenet.load_model(celeb_model)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        feed_dict = {images_placeholder: prewhitened, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        classifier_filename_exp = os.path.expanduser(classifier)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')
        print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)
        predictions = model.predict_proba(emb)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        result_string = ""
        for i in range(count):
            result_string += class_names[best_class_indices[i]]+', '
        print(result_string)
        return result_string


pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'


def get_args():

    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")

    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--classifier',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    image_dir = Path(image_dir)

    for image_path in image_dir.glob("*.*"):
        img = cv2.imread(str(image_path), 1)
        print(type(img))

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file
    global margin
    margin = args.margin
    image_dir = args.image_dir
    global celeb_model
    celeb_model = args.model
    global classifier
    classifier = args.classifier

    if not weight_file:
        weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                               file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    global detector
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    global model
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)
    model._make_predict_function()


if __name__ == '__main__':
    main()
    app.run(host='143.248.36.213', port=3355, debug=True)
