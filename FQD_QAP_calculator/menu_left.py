from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QGridLayout,
    QSpinBox, QTabWidget, QRadioButton,
)


class Main_Left(QVBoxLayout):
    def __init__(self, parent):
        super().__init__()
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)

        self.tabs.addTab(FQD_Tab(parent), "FQD")
        self.tabs.addTab(QAP_Tab(parent), "QAP")
        self.addWidget(self.tabs)


class FQD_Tab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(QLabel("Choose FQD type"))
        self.qual_checkbox = QCheckBox("Qualitative")
        self.quant_checkbox = QCheckBox("Quantitative")
        self.layout.addWidget(self.qual_checkbox)
        self.layout.addWidget(self.quant_checkbox)
        self.layout.addSpacing(20)
        self.layout.addWidget(QLabel("Fragments size range"))
        size_layout = QGridLayout()
        size_layout.addWidget(QLabel("Min"), 0, 0)
        size_layout.addWidget(QLabel("Max"), 1, 0)
        self.min_spin = QSpinBox()
        self.max_spin = QSpinBox()
        self.min_spin.setRange(2, 9)
        self.max_spin.setRange(2, 9)
        self.max_spin.setValue(9)
        self.min_spin.valueChanged.connect(self.spin_min_changed)
        self.max_spin.valueChanged.connect(self.spin_max_changed)
        size_layout.addWidget(self.min_spin, 0, 1)
        size_layout.addWidget(self.max_spin, 1, 1)
        self.layout.addLayout(size_layout)
        self.layout.addSpacing(20)
        calculate_button = QPushButton("Calculate and save")
        self.layout.addWidget(calculate_button)
        self.layout.addStretch(1)

        calculate_button.clicked.connect(self.calculate)

    def calculate(self):
        curr_tab = self.parent.tabs.currentWidget()
        tab_name = self.parent.get_tab_type()
        if tab_name == "excel":
            if self.parent.read_file is None:
                QMessageBox.critical(self.parent, "Error", "Read file with SMILES before calculation.")
                return
            else:
                smiles_col = curr_tab.combobox.currentText()
                smiles = self.parent.read_file[smiles_col].tolist()
        elif tab_name == "smiles":
            self.parent.read_text = curr_tab.read_text()
            if self.parent.read_text is None:
                QMessageBox.critical(self.parent, "Error", "Put SMILES as text.")
                return
            else:
                smiles = self.parent.read_text
        else:
            QMessageBox.critical(self.parent, "Error", "Choose other tab.")
            return None
        qual = self.qual_checkbox.checkState() == Qt.CheckState.Checked
        quant = self.quant_checkbox.checkState() == Qt.CheckState.Checked
        if qual and quant:
            fqd_type = "all"
        elif qual:
            fqd_type = "qual"
        elif quant:
            fqd_type = "quant"
        else:
            QMessageBox.critical(self.parent, "Error", "Choose FQD type.")
            return None

        import fqd
        fqds_range = range(self.min_spin.value(), self.max_spin.value() + 1)

        fqds = fqd.generate_features(
            smiles,
            fqd_type=fqd_type,
            atoms_range=fqds_range
            )
        filename = QFileDialog.getSaveFileName(self.parent)[0]
        if filename.endswith(".csv"):
            fqds.to_csv(filename)
        else:
            fqds.to_csv(f"{filename}.csv")

    def spin_min_changed(self, i):
        self.max_spin.setMinimum(i)

    def spin_max_changed(self, i):
        self.min_spin.setMaximum(i)

class QAP_Tab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(QLabel("Choose QAP type"))
        self.qap_type = "sum"
        self.layout.addSpacing(20)

        radiobutton = QRadioButton("sumQAP")
        radiobutton.setChecked(True)
        radiobutton.qap_type = "sum"
        radiobutton.toggled.connect(self.onClicked)
        self.layout.addWidget(radiobutton)

        radiobutton = QRadioButton("histQAP")
        radiobutton.qap_type = "hist"
        radiobutton.toggled.connect(self.onClicked)
        self.layout.addWidget(radiobutton)

        radiobutton = QRadioButton("spQAP")
        radiobutton.qap_type = "boomed"
        radiobutton.toggled.connect(self.onClicked)
        self.layout.addWidget(radiobutton)

        self.layout.addSpacing(20)

        calculate_button = QPushButton("Calculate and save")
        self.layout.addWidget(calculate_button)
        self.layout.addStretch(1)

        calculate_button.clicked.connect(self.calculate)

    def onClicked(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.qap_type = radioButton.qap_type


    def calculate(self):
        curr_tab = self.parent.tabs.currentWidget()
        tab_name = self.parent.get_tab_type()
        if tab_name == "excel":
            if self.parent.read_file is None:
                QMessageBox.critical(self.parent, "Error", "Read file with SMILES before calculation.")
                return
            else:
                smiles_col = curr_tab.combobox.currentText()
                smiles = self.parent.read_file[smiles_col].tolist()
        elif tab_name == "smiles":
            self.parent.read_text = curr_tab.read_text()
            if self.parent.read_text is None:
                QMessageBox.critical(self.parent, "Error", "Put SMILES as text.")
                return
            else:
                smiles = self.parent.read_text
        else:
            QMessageBox.critical(self.parent, "Error", "Choose other tab.")
            return None

        import apairs_descriptor_calculator
        if self.qap_type == "hist":
            qaps = apairs_descriptor_calculator.generate_apairs_hist_descriptors(smiles)
        elif self.qap_type == "boomed" or self.qap_type == "sum":
            qaps = apairs_descriptor_calculator.generate_apairs_descriptors(smiles, desc_type=self.qap_type)
        else:
            QMessageBox.critical(self.parent, "Error", "Choose QAP type.")
            return None

        filename = QFileDialog.getSaveFileName(self.parent)[0]
        qaps.drop(columns=["mol", "apairs", "apairs_applicability"], inplace=True)
        if filename.endswith(".csv"):
            qaps.to_csv(filename)
        else:
            qaps.to_csv(f"{filename}.csv")
