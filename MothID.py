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

family_model   = "mothfamily-20171004.pb"
family_labels  = "family.txt"
genus_model    = "mothgenus-20171007.pb"
genus_labels   = "genus.txt"
species_model  = "mothspecies-20171005.pb"
species_labels = "species.txt"
input_height   = 299
input_width    = 299
input_mean     = 0
input_std      = 255
input_layer    = "input"
output_layer   = "InceptionV3/Predictions/Reshape_1"

use_family  = True
use_genus   = False
use_species = True

def classify_image(file_name, return_filename=False):
  t = read_tensor_from_image_file(file_name, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer

  return_value = ""
  return_array = []

  if(use_family):
    input_operation = family_graph.get_operation_by_name(input_name);
    output_operation = family_graph.get_operation_by_name(output_name);
    with tf.Session(graph=family_graph) as sess:
      results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    results = np.squeeze(results)
    predictions = results.argsort()[::-1]
    prediction = predictions[0]

    return_value += "Family: %s (%d%%)\n" % (family_labels[prediction], results[prediction] * 100)
    return_array.append("%s (%d%%)" % (family_labels[prediction], results[prediction] * 100))
    prefix = family_labels[predictions[0]] + " "

  if(use_genus):
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
    return_value += "  Genus: %s (%d%%, %d%%)\n" % (genus_labels[prediction].split()[1], results[prediction] * 100, (results[prediction] / total) * 100)
    return_array.append("%s (%d%%, %d%%)" % (genus_labels[prediction].split()[1], results[prediction] * 100, (results[prediction] / total) * 100))
    prefix = genus_labels[filtered[0]] + " "

  if(use_species):
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
    return_array.append("%s (%d%%, %d%%)" % (species_labels[i].split(' ', 1)[1], results[0] * 100, (results[0] / total) * 100))
    for i in filtered:
        return_value += "    %s (%d%%, %d%%)\n" % (species_labels[i].split(' ', 1)[1], results[i] * 100, (results[i] / total) * 100)

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
        self.displayAndClassifyImage(path)

    def initUI(self):
        self.setWindowTitle("Moth ID")
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

    def displayAndClassifyImage(self, path):
        self.label.setImage(path)
        self.resizeUI()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.text.setText(classify_image(path, false))
        QApplication.restoreOverrideCursor()
        return

    def classifyImage(self):
        path = QFileDialog.getOpenFileName(self, "Choose an image", "", "Images (*.bmp *.gif *.jpg *.jpeg *.png)")
        if(path[0]):
            self.displayAndClassifyImage(path[0])
        return

    def classifyDirectory(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a directory", "", QFileDialog.ShowDirsOnly)
        if(path):
            result_path = os.path.join(path, "results")
            if not os.path.isdir(result_path):
                os.mkdir(result_path)

            print(os.path.join(path, "*.{bmp,gif,jpg,jpeg,png}"))
            files = glob.glob(os.path.join(path, "*.bmp"))
            files.extend(glob.glob(os.path.join(path, "*.gif")))
            files.extend(glob.glob(os.path.join(path, "*.jpg")))
            files.extend(glob.glob(os.path.join(path, "*.jpeg")))
            files.extend(glob.glob(os.path.join(path, "*.png")))
            print(len(files))

            progdialog = QProgressDialog("", "Cancel", 0, len(files), self)
            progdialog.setWindowTitle("Classifying")
            progdialog.setWindowModality(Qt.WindowModal)
            progdialog.show()

            i = 0
            for file in files:
                if progdialog.wasCanceled():
                    break

                result = classify_image(file, True)

                file_name, file_ext = os.path.splitext(os.path.basename(file))
                destination = os.path.join(result_path, file_name + " - " + result + file_ext)
                shutil.copyfile(file, destination)

                progdialog.setValue(i)
                i += 1

            progdialog.close()

        return

if __name__ == '__main__':
    family_graph = load_graph(family_model)
    genus_graph = load_graph(genus_model)
    species_graph = load_graph(species_model)

    family_labels = load_labels(family_labels)
    genus_labels = load_labels(genus_labels)
    species_labels = load_labels(species_labels)

    app = QApplication(sys.argv)
    win = MothID()
    sys.exit(app.exec_())
