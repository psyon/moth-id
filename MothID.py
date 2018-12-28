from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import glob

import numpy as np
import tensorflow as tf

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

input_height   = 299
input_width    = 299
input_mean     = 0
input_std      = 255
input_layer    = "input"
output_layer   = "InceptionV3/Predictions/Reshape_1"

use_family  = False
use_genus   = False
use_species = False

def _most_recent_model(prefix):
    files = glob.glob("models/" + prefix +  "*.pb")
    if len(files) > 0:
        files.sort(reverse=True)
        labels = files[0].replace(".pb", ".txt");
        return files[0], labels
    return None,None

def classify_image(file_name, return_filename=False):
    t = read_tensor_from_image_file(file_name, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    return_value = ""
    return_array = []

    prefix = ""
    indent = ""

    if use_family:
        input_operation = family_graph.get_operation_by_name(input_name);
        output_operation = family_graph.get_operation_by_name(output_name);
        with tf.Session(graph=family_graph) as sess:
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        predictions = results.argsort()[::-1]
        prediction = predictions[0]

        return_value += "%s (%d%%)\n" % (family_labels[prediction], results[prediction] * 100)
        return_array.append("%s (%d%%)" % (family_labels[prediction], results[prediction] * 100))
        prefix = family_labels[predictions[0]] + " "
        indent = "    "

    if use_genus:
        input_operation = genus_graph.get_operation_by_name(input_name);
        output_operation = genus_graph.get_operation_by_name(output_name);
        with tf.Session(graph=genus_graph) as sess:
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        predictions = results.argsort()[::-1]

        total = 0
        filtered = []
        for i in predictions:
            if(genus_labels[i].startswith(prefix)):
                total += results[i]
                filtered.append(i)

        prediction = filtered[0]
        return_value += "%s%s (%d%%, %d%%)\n" % (indent, genus_labels[prediction].split()[1], results[prediction] * 100, (results[prediction] / total) * 100)
        return_array.append("%s (%d%%, %d%%)" % (genus_labels[prediction].split()[1], results[prediction] * 100, (results[prediction] / total) * 100))
        prefix = genus_labels[filtered[0]] + " "
        indent = indent + "    "

    if use_species:
        input_operation = species_graph.get_operation_by_name(input_name);
        output_operation = species_graph.get_operation_by_name(output_name);
        with tf.Session(graph=species_graph) as sess:
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        predictions = results.argsort()[::-1]

        total = 0
        filtered = []
        for i in predictions:
            if(species_labels[i].startswith(prefix)):
                total += results[i]
                filtered.append(i)

        filtered = filtered[:5]

        if len(filtered) > 0:
            name = species_labels[filtered[0]].replace(prefix, "")
            return_array.append("%s (%d%%, %d%%)" % (name, results[filtered[0]] * 100, (results[filtered[0]] / total) * 100))

            for i in filtered:
                name = species_labels[i].replace(prefix, "")
                return_value += "%s%s (%d%%, %d%%)\n" % (indent, name, results[i] * 100, (results[i] / total) * 100)

        if(return_filename):
            return ' - '.join(return_array)

    return return_value

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)

    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3, name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

class ClassifyDirectoryThread(QThread):
    count = pyqtSignal(int)
    display = pyqtSignal('QString')
    result = pyqtSignal('QString')
    progress = pyqtSignal(int)
    complete = pyqtSignal()

    def __init__(self, path):
        QThread.__init__(self)
        self.path = path
        self.running = True

    def __del__(self):
        self.wait()

    def stop(self):
        self.running = False

    def run(self):
        result_path = os.path.join(self.path, "results")
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        files = glob.glob(os.path.join(self.path, "*.bmp"))
        files.extend(glob.glob(os.path.join(self.path, "*.gif")))
        files.extend(glob.glob(os.path.join(self.path, "*.jpg")))
        files.extend(glob.glob(os.path.join(self.path, "*.jpeg")))
        files.extend(glob.glob(os.path.join(self.path, "*.png")))

        self.count.emit(len(files))

        i = 0
        for file in files:
            if not self.running:
                break;

            result = classify_image(file, True)
            self.display.emit(file)
            self.result.emit(result)
            self.progress.emit(i)

            file_name, file_ext = os.path.splitext(os.path.basename(file))
            destination = os.path.join(result_path, file_name + " - " + result + file_ext)
            shutil.copyfile(file, destination)
            i += 1

        self.complete.emit()

class ClassifyImageThread(QThread):
    result = pyqtSignal('QString')
    complete = pyqtSignal()

    def __init__(self, path):
        QThread.__init__(self)
        self.path = path

    def __del__(self):
        self.wait()

    def run(self):
        result = classify_image(self.path, False)
        self.result.emit(result)
        self.complete.emit()

class ImageLabel(QLabel):
    def __init__(self, image):
        super(ImageLabel, self).__init__("")
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setImage(image)

    def setImage(self, image):
        self.pixmap = QPixmap(image).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.pixmap)

    def width(self):
        return self.pixmap.width()

    def height(self):
        return self.pixmap.height()

class MothID(QMainWindow):
    progdialog = None
    thread = None

    def __init__(self):
        super(MothID, self).__init__()
        self.setAcceptDrops(True)
        self.initUI()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        path = str(e.mimeData().urls()[0].toLocalFile())

        if os.path.isdir(path):
            self.classifyDirectory(path)
        else:
            self.displayAndClassifyImage(path)

    def initUI(self):
        self.setWindowTitle("Moth ID")
        self.setWindowIcon(QIcon('moth.ico'))
        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        fileAct = QAction('Classify &Image', self)
        fileAct.setShortcut('Ctrl+I')
        fileAct.setStatusTip('Classify a single image')
        fileAct.triggered.connect(self.classifyImage)

        directoryAct = QAction('Classify &Directory', self)
        directoryAct.setShortcut('Ctrl+D')
        directoryAct.setStatusTip('Classify all images in a directory')
        directoryAct.triggered.connect(self.classifyDirectory)

        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(fileAct)
        fileMenu.addAction(directoryAct)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAct)

        self.label = ImageLabel("drop.png");
        layout.addWidget(self.label)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setAcceptDrops(False)
        layout.addWidget(self.text)

        self.resizeUI()

        rect = self.frameGeometry()
        center = QDesktopWidget().availableGeometry().center()
        rect.moveCenter(center)
        self.move(rect.topLeft())

        self.show()

    def resizeUI(self):
        self.resize(self.label.width(), self.label.height() + self.menuBar().height() + + 150);

    def displayImage(self, path):
        self.label.setImage(path)
        self.resizeUI()

    def displayAndClassifyImage(self, path):
        self.displayImage(path)
        self.text.setText('')

        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.thread = ClassifyImageThread(path)
        self.thread.result.connect(self.signalResult)
        self.thread.complete.connect(self.signalComplete)
        self.thread.start()
        return

    def signalCancel(self):
        if self.thread:
            self.thread.stop()
            self.thread.wait()
            self.thread = None
        if self.progdialog:
            self.progdialog.close()
            self.progdialog = None

    def signalImageCount(self, count):
        if self.progdialog:
            self.progdialog.setMaximum(count)

    def signalProgress(self, value):
        if self.progdialog:
            self.progdialog.setValue(value)

    def signalComplete(self):
        if self.progdialog:
            self.progdialog.close()
        QApplication.restoreOverrideCursor()

    def signalResult(self, result):
        self.text.setText(result)

    def classifyImage(self):
        path = QFileDialog.getOpenFileName(self, "Choose an image", "", "Images (*.bmp *.gif *.jpg *.jpeg *.png)")
        if(path[0]):
            self.displayAndClassifyImage(path[0])

    def classifyDirectory(self, path):
        self.progdialog = QProgressDialog("", "Cancel", 0, 100, self)
        self.progdialog.setWindowTitle("Classifying")
        self.progdialog.setWindowModality(Qt.WindowModal)
        self.progdialog.canceled.connect(self.signalCancel)
        self.progdialog.show()

        self.thread = ClassifyDirectoryThread(path)
        self.thread.count.connect(self.signalImageCount)
        self.thread.display.connect(self.displayImage)
        self.thread.result.connect(self.signalResult)
        self.thread.progress.connect(self.signalProgress)
        self.thread.complete.connect(self.signalComplete)
        self.thread.start()

    def menuClassifyDirectory(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a directory", "", QFileDialog.ShowDirsOnly)
        if(path):
            self.classifyDirectory(path)

if __name__ == '__main__':
    family_model, family_labels = _most_recent_model("mothfamilies")
    genus_model, genus_labels = _most_recent_model("mothgenus")
    species_model, species_labels = _most_recent_model("mothspecies")

    if family_model:
        use_family = True
        family_graph = load_graph(family_model)
        family_labels = load_labels(family_labels)

    if genus_model:
        use_genus = True
        genus_graph = load_graph(genus_model)
        genus_labels = load_labels(genus_labels)

    if species_model:
        use_species = True
        species_graph = load_graph(species_model)
        species_labels = load_labels(species_labels)

    app = QApplication(sys.argv)
    win = MothID()
    sys.exit(app.exec_())
