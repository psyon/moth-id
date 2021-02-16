from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import glob

import numpy as np

import tflite_runtime.interpreter as tflite
from PIL import Image

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

input_height   = 299
input_width    = 299
input_mean     = 0
input_std      = 255
input_layer    = "input"
output_layer   = "InceptionV3/Predictions/Reshape_1"

def _most_recent_model():
    files = glob.glob("models/*.tflite")
    if len(files) > 0:
        files.sort(reverse=True, key=lambda name: name[-11:])
        labels = files[0].replace(".tflite", ".txt");
        return files[0], labels
    return None,None

def classify_image(file_name, return_filename=False):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32

    # Load the image
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(file_name).resize((width, height))

    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    predictions = results.argsort()[-5:][::-1]

    if(return_filename):
        p = predictions[0]
        return "%s (%d%%)" % (species_labels[p], results[p] * 100)

    return_value = ""
    for i in predictions:
        return_value += "%s (%d%%)\n" % (species_labels[i], results[i] * 100)

    return return_value

def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

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

    def closeEvent(self, e):
        self.quit()

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

    def quit(self):
        app.quit()
        sys.exit()

    def initUI(self):
        self.setWindowTitle("Moth ID - " + os.path.basename(species_model))
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
        directoryAct.triggered.connect(self.menuClassifyDirectory)

        chooseAct = QAction('Choose &Model', self)
        chooseAct.setShortcut('Ctrl+M')
        chooseAct.setStatusTip('Choose a new model file for classifications.')
        chooseAct.triggered.connect(self.chooseModelFile)

        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(self.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(fileAct)
        fileMenu.addAction(directoryAct)
        fileMenu.addSeparator()
        fileMenu.addAction(chooseAct)
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

    def chooseModelFile(self):
        global species_model, species_graph, species_labels, interpreter
        path = QFileDialog.getOpenFileName(self, "Choose a model file", "./models", "Models (*.tflite)")
        if(path[0]):
            file_model  = path[0]
            file_labels = path[0].replace(".tflite", ".txt");
            if os.path.exists(file_labels):
                species_model = file_model
                interpreter = tflite.Interpreter(model_path=species_model)
                interpreter.allocate_tensors()
                species_labels = load_labels(file_labels)
                self.setWindowTitle("Moth ID - " + os.path.basename(species_model))
            else:
                QMessageBox.warning(self, "Error", "Label file not found.")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    species_model, species_labels = _most_recent_model()
    if species_model == None:
        QMessageBox.critical(None, "Error", "Could not find default model file.")
        sys.exit(-1)

    interpreter = tflite.Interpreter(model_path=species_model)
    interpreter.allocate_tensors()

    species_labels = load_labels(species_labels)

    win = MothID()
    sys.exit(app.exec_())
