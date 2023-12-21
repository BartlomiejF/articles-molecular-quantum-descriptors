import sys
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QTabWidget,
    QWidget,
    QCheckBox,
    QComboBox,
    QTextEdit,
    QFileDialog,
    QMessageBox,
)
from menu_left import Main_Left
from enum import Enum




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FQD & QAP calculator")
        layout_outer = QVBoxLayout()
        layout_inner = QHBoxLayout()
        layout_inner.addLayout(Main_Left(self))

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)

        self.tabs.addTab(Excel_Csv_File(self), "Excel/CSV file")
        self.tabs.addTab(Smiles_Text(), "SMILES text")
        self.tabs.addTab(Info(), "Info")

        layout_inner.addWidget(self.tabs)
        layout_outer.addLayout(layout_inner)
        self.setLayout(layout_outer)

        w = QWidget()
        w.setLayout(layout_outer)
        self.setCentralWidget(w)

        self.read_file = None
        self.read_text = None

    def get_tab_type(self):
        curr_tab = self.tabs.currentWidget()
        if type(curr_tab) == Excel_Csv_File:
            return "excel"
        elif type(curr_tab) == Smiles_Text:
            return "smiles"
        else:
            return "info"


class Excel_Csv_File(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.parent = parent

        readfile_button = QPushButton("Read file")
        self.layout.addWidget(readfile_button)
        self.layout.addSpacing(20)
        self.layout.addWidget(QLabel("Choose SMILES column"))
        self.combobox = QComboBox()
        self.layout.addWidget(self.combobox)
        self.layout.addStretch(1)
        readfile_button.clicked.connect(self.read)

    def read(self):
        file = QFileDialog(self).getOpenFileName()[0]
        if file.endswith("csv"):
            self.parent.read_file = pd.read_csv(file)
        elif file.endswith("xls") or file.endswith("xlsx"):
            self.parent.read_file = pd.read_excel(file)
        else:
            QMessageBox.critical(self, "Error", "The file must be either excel or csv.")
            return None
        self.combobox.addItems(self.parent.read_file.columns.tolist())


class Smiles_Text(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel("Add SMILES text:"))
        self.smiles_text = QTextEdit()
        layout.addWidget(self.smiles_text)
        layout.addStretch(1)

    def read_text(self):
        text = self.smiles_text.toPlainText()
        text = text.split("\n")
        text = [x.strip(" ,") for x in text if len(x)>0]
        if len(text) > 0:
            return text
        else:
            return None


class Info(QWidget):
    def __init__(self):
        super().__init__()
        text = """Usage:

Excel/CSV tab:
1. Read excel or csv file that contains SMILES in one of columns.
2. Set on the dropdown menu the name of column that contains SMILES.
3. Choose types of descriptors you would like to calculate.
4. Push calculate and save button.
5. Save .csv file with calculated descriptors.

SMILES text tab:
1. Paste SMILES into the text window. 1 SMILES - 1 line.
2. Choose types of descriptors you would like to calculate.
3. Push calculate and save button.
4. Save .csv file with calculated descriptors.

Contact: bartlomiej.fliszkiewicz@wat.edu.pl

Link to [GitHub](https://github.com/BartlomiejF) 
"""
        layout = QVBoxLayout()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        label.setTextFormat(Qt.MarkdownText)
        layout.addWidget(label)
        layout.addStretch(1)
        self.setLayout(layout)



app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()