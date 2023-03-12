from asyncio.windows_events import NULL
import streamlit as st
import base64
from skimage import io
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import mysql.connector
from sklearn.model_selection import train_test_split
import pathlib
import math


# Download performance metrics
img_tes = []
list = []
if 'w_hidden' not in st.session_state:
    st.session_state['w_hidden'] = []
if 'b_hidden' not in st.session_state:
    st.session_state['b_hidden'] = []
if 'w_output' not in st.session_state:
    st.session_state['w_output'] = []
if 'max_skl' not in st.session_state:
    st.session_state['max_skl'] = []
if 'min_skl' not in st.session_state:
    st.session_state['min_skl'] = []


# Untuk mengkoneksikan kedatabase mysql


def opendb():
    # cursor untuk mengeksekusi perintah disql/query
    global mydb, cursor
    mydb = mysql.connector.connect(
        user='root',
        password='',
        database='botol_ta',
        host='127.0.0.1'
    )
    cursor = mydb.cursor()

# Untuk Menutup koneksi dari database mysql


def closedb():
    global mydb, cursor
    cursor.close()
    mydb.close()


# def hstg(img, label, filename):
#     path = img.split('.')[0][-2:]
#     paths = "D:/Pendadaran/Acak/"

#     image = cv2.imread(paths + img)
#     color = ['r', 'g', 'b']
#     data = []
#     for i, col in enumerate(color):
#         histg = cv2.calcHist([image], [i], None, [256], [0, 256])
#         persen = persentase(histg)
#         plt.plot(histg, color=col)
#         plt.xlim([0, 256])
#         data.append(np.average(persen[0:255]))
#         # data.append(np.average(histg))

#     plt.plot(histg)
#     # plt.show()
#     img_tes = histg
#     print("hasil data")
#     print(data)
#     return data


def get_RGB2(img1):

    img = io.imread("D:/Pendadaran/seg_pred/"+img1)[:, :, :]
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    # print(dominant)
    return dominant


def get_RGB(img1):
    img = io.imread(img1)[:, :, :]
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    # print(dominant)
    return dominant


# def histogram(img, label, filename):
#     color = ['b', 'g', 'r']
#     data = []
#     rgb = []
#     for i, col in enumerate(color):
#         histg = cv2.calcHist([img], [i], None, [256], [0, 256])
#         persen = persentase(histg)
#         plt.plot(histg, color=col)
#         plt.xlim([0, 256])
#         data.append(np.average(persen[0:255]))
#         rgb = histg
#         # st.info(color[i])
#         # st.info(filename)
#         # st.bar_chart(histg)
#     data.append(label)
#     data.append(filename)
#     plt.plot(histg)
#     return data

# persentase data histogram dari citra


# def persentase(histro):
#     persen = []
#     for i in range(len(histro)):
#         persen.append((histro[i]/sum(histro))*100)
#     return persen


def loadImages(path):
    # untuk meng-list nama2 file yang ada di folder
    temp = os.listdir(path)
    data = []

    for i in temp:
        # endswith berfungsi memilih string yang akhirannya png
        if (i.endswith("jpg")):
            data.append(path+i)

    return(data)

# Input Database


def data0():
    img = loadImages("D:/Pendadaran/seg_train/buildings//")

    for i in range(len(img)):
        image = cv2.imread(img[i])
        file = pathlib.Path(str(img[i])).name
        filename = file.translate(str.maketrans({"]": None}))
        # merubah data citra menjadi nilai histogram
        rgb = get_RGB(img[i])

        data = (str(rgb[0]), str(rgb[1]), str(rgb[2]), '0', filename)

        opendb()
        cursor.execute(
            '''insert into botol(b,g,r, label, file) values('%s', '%s', '%s', '%s', '%s')''' % data)
        # untuk melakukan perubahan pada tabel.
        mydb.commit()
        closedb()
        print('BGR :', data, "-", i)


def data1():
    img = loadImages("D:/Pendadaran/seg_train/glacier//")

    for i in range(len(img)):
        image = cv2.imread(img[i])
        file = pathlib.Path(str(img[i])).name
        filename = file.translate(str.maketrans({"]": None}))
        rgb = get_RGB(img[i])

        data = (str(rgb[0]), str(rgb[1]), str(rgb[2]), "1", filename)

        opendb()
        cursor.execute(
            '''insert into botol(b,g,r, label, file) values('%s', '%s', '%s', '%s', '%s')''' % data)
        mydb.commit()
        closedb()

        print('BGR :', data, "-", i)


def selectData(target):
    opendb()

    query = "SELECT * FROM `botol` WHERE label = %s" % (target)
    cursor.execute(query)
    records = cursor.fetchall()

    return records


def run_all():
    with st.spinner("Wait for it...."):
        data0()
        data1()
    st.success("RGB Done!")
 # fungi sigmoid


def sigmoid(x):
    return 1/(1+math.exp(-x))


tr_acc = 0
tr_error = 0
tr_epoch = 0
tr_epochs = 0
tr_target = []

tr_0_hidden = np.empty([3, 4])
tr_o_output = np.empty([3, 4])
tr_new_target = []


# Training Data dan Backpropagation


def training(index=0, show=0):
    global in_hidden
    global in_lr
    global in_epoch

    if in_hidden == "":
        num_hidden = 3
    else:
        num_hidden = int(in_hidden)

    if in_lr == "":
        lr = 0.1
    else:
        lr = float(in_lr)

    if in_epoch == "":
        epochs = 100
    else:
        epochs = int(in_epoch)

    allData_0 = []
    label_0 = []
    allData_1 = []
    label_1 = []
    nama_file0 = []
    nama_file1 = []
    data_0 = selectData(0)
    data_1 = selectData(1)

    for i in data_0:
        f = [i[1], i[2], i[3]]
        allData_0.append(f)
        label_0.append(i[4])
        nama_file0.append(i[5])

    for i in data_1:
        f = [i[1], i[2], i[3]]
        allData_1.append(f)
        label_1.append(i[4])
        nama_file1.append(i[5])

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(
        allData_0, label_0, test_size=0.2, shuffle=True)
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        allData_1, label_1, test_size=0.2, shuffle=True)

    sig = np.vectorize(sigmoid)

    # data d bawah untuk menggabungkan data latih dan uji
    data_train = np.concatenate((X_train_0, X_train_1), axis=0)
    data_test = np.concatenate((X_test_0, X_test_1), axis=0)
    label_train = np.concatenate((y_train_0, y_train_1), axis=0)
    label_test = np.concatenate((y_test_0, y_test_1), axis=0)
    nama_file = np.concatenate((nama_file0, nama_file1), axis=0)

    # input
    # NORMALISASI
    input = np.array(data_train, dtype='float64')
    skl = MinMaxScaler()
    skl.fit(input)
    max_skl = skl.data_max_
    min_skl = skl.data_min_
    st.session_state['max_skl'] = max_skl
    st.session_state['min_skl'] = min_skl

    input_skl = skl.transform(input)
    print("INPUT = ", input_skl)
    # output
    target = to_categorical(label_train)
    # configurasi (jumlah perceptron per layer)
    num_input = input.shape[1]
    num_hidden = num_hidden
    num_output = target.shape[1]
    # hidden layer
    w_hidden = np.random.uniform(low=-1, high=1, size=(num_input, num_hidden))
    b_hidden = np.random.uniform(low=-1, high=1, size=(num_hidden))

    # output layeer
    w_output = np.random.uniform(low=-1, high=1, size=(num_hidden, num_output))
    b_output = np.random.uniform(low=-1, high=1, size=(num_output))

    # simpan loss dan accuracy
    loss_values = []
    acc_values = []
    n = 1
    lr = lr
    epochs = epochs
    for epoch in range(epochs):
        MSE = 0
        new_target = np.zeros(target.shape)
        n = n+1
        for idx, inp in enumerate(input_skl):
            # batch
            # --feedforward

            o_hidden = np.matmul(input_skl[idx], w_hidden) + b_hidden
            o_hidden = sig(o_hidden)

            o_output = np.matmul(o_hidden, w_output)
            o_output = sig(o_output)

            # ----error
            error = target[idx] - o_output

            MSE = MSE + (np.sum(error**2))
            new_target[idx] = o_output.round()

            # --backpropagation
            eh = error @ w_output.T
            w_output = w_output + \
                (lr * ((error * o_output * (1 - o_output))
                 * o_hidden[np.newaxis].T))
            w_hidden = w_hidden + \
                (lr * ((eh * o_hidden * (1-o_hidden))
                 * input_skl[idx][np.newaxis].T))

            b_output = b_output + (lr * ((error * o_output * (1 - o_output))))
            b_hidden = b_hidden + (lr * ((eh * o_hidden * (1-o_hidden))))
            n = n+1

        MSE = MSE/n
        acc = 1-(np.sum(np.absolute(target - new_target))/n)

        acc_values.append(acc)
        loss_values.append(MSE)
        acc = acc
        print("Epoch ", epoch, "/", epochs, ": error", MSE, " accuracy: ", acc)
        #print(target, "----", new_target)

       # glob utk menu
    global tr_0_hidden
    tr_0_hidden = o_hidden
    global tr_b_hidden
    st.session_state['b_hidden'] = b_hidden

    tr_b_hidden = b_hidden
    global tr_o_output
    tr_o_output = o_output
    global tr_target
    tr_target = target
    global tr_w_hidden
    tr_w_hidden = w_hidden
    st.session_state['w_hidden'] = w_hidden
    global tr_w_output
    tr_w_output = w_output
    st.session_state['w_output'] = w_output
    global tr_new_target
    tr_new_target = new_target

    global tr_acc
    tr_acc = acc
    global tr_error
    tr_error = MSE
    global tr_epoch
    tr_epoch = epoch
    global tr_epochs
    tr_epochs = epochs
    print("bias Hidden Neuron Pelatihan")
    print(b_hidden)
    print("Bobot Hidden Neuron Pelatihan")
    print(w_hidden)
    print("Bobot Output Neuron Pelatihan")
    print(w_output)
    return [b_hidden, w_hidden, w_output]

# GEt indeks Database


def selectIndex(target):
    opendb()

    query = "SELECT `id` FROM `botol` WHERE file = '" + str(target) + "'"
    cursor.execute(query)
    records = cursor.fetchone()
    # st.title(records)
    return records


def ambil_bobot(gambar):

    img_hist = get_RGB2(gambar)
    # print(img_hist)
    max_rgb = st.session_state['max_skl']
    min_rgb = st.session_state['min_skl']
    cek_histg = []
    normal = (img_hist-min_rgb)/(max_rgb-min_rgb)

    print(normal)
    path = gambar.split('.')[0][-2:]

    w_hidden_akhir = st.session_state['w_hidden']
    b_hidden_akhir = st.session_state['b_hidden']
    w_output_akhir = st.session_state['w_output']
    print("bias Hidden Neuron test")
    print(b_hidden_akhir)
    print("Bobot Hidden Neuron test")
    print(w_hidden_akhir)
    print("Bobot Output Neuron test")
    print(w_output_akhir)
    sig = np.vectorize(sigmoid)
    print("sig = ", sig)
    o_hidden = np.matmul(normal, w_hidden_akhir) + b_hidden_akhir
    o_hidden = sig(o_hidden)
    print("output hidden = ", o_hidden)
    o_output = np.matmul(o_hidden, w_output_akhir)
    o_output = sig(o_output)
    print("output output = ", o_output)
    new_target_test = o_output.round()
    test_target = new_target_test
    print("hasil input jaringan")
    print(test_target)
    max_index_row = np.argmax(test_target)
    print(max_index_row)
    if max_index_row == 0:
        st.title("Kelas Botol Mineral")
        # st.text(cek_histg)
    else:
        st.title("Kelas Bukan Botol Mineral")
        # st.text(cek_histg)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MENU MENU MENU
# =====================================================
    # Sidebar - Header
st.sidebar.title("Klasifikasi Botol Mineral")

st.sidebar.caption("Preparing Data Latih")
if st.sidebar.button("GET RGB Training"):
    run_all()
    st.sidebar.header("Sukses")

st.sidebar.caption("Setting Training")
in_hidden = st.sidebar.text_input("Hidden Layer")
in_lr = st.sidebar.text_input("Leraning Rate")
in_epoch = st.sidebar.text_input("Epoch")

st.sidebar.caption("Jalankan Training")
if st.sidebar.button("Training"):
    with st.spinner("Wait for it...."):
        list = training()
        print(list)
    st.success("Training Done!")
    st.sidebar.text("Akurasi")
    st.sidebar.info(tr_acc)
    st.sidebar.text("Error")
    st.sidebar.info(tr_error)


st.sidebar.header('TES Kelas Botol')
uploaded_file = st.sidebar.file_uploader(
    'Upload your image file', type=['JPG'],)

# Main PAnel
st.markdown("""
<style>
.big-font {
    font-size:40px !important;
    align:center;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Kelasifikasi Sampah Botol Mineral Backpropagation</p>',
            unsafe_allow_html=True)


def akurasi(url):
    st.markdown(
        '<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)


if uploaded_file is not None:
    st.image(uploaded_file, width=500, output_format="200")
    with st.spinner("Wait for it...."):
        # time.sleep(5)
        hs_tes = get_RGB2(uploaded_file.name)
        # st.text(hs_tes)
        ambil_bobot(uploaded_file.name)


else:
    st.info('Menunggu Inputan Gambar')
