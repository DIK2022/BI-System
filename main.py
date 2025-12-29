# main.py
import sys
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

# Импортируем кастомные модули
from data_models import PandasTableModel, PolarsTableModel, SortFilterProxyModel
from workers import DataLoaderThread, ExportWorker
from visualization import (
    MatplotlibWidget, 
    PlotlyWidget, 
    PyQtGraphWidget,
    create_plotly_figure
)
from utils import DataProcessor, DataConverter, StyleManager


class DataViewWidget(QWidget):
    """Виджет для отображения табличных данных"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Панель инструментов
        toolbar = QToolBar()
        self.sort_asc_btn = QAction("Сортировка ↑", self)
        self.sort_desc_btn = QAction("Сортировка ↓", self)
        self.filter_btn = QAction("Фильтр", self)
        self.clear_filter_btn = QAction("Очистить", self)
        
        toolbar.addAction(self.sort_asc_btn)
        toolbar.addAction(self.sort_desc_btn)
        toolbar.addSeparator()
        toolbar.addAction(self.filter_btn)
        toolbar.addAction(self.clear_filter_btn)
        
        # Таблица
        self.table_view = QTableView()
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        
        # Прокси модель для фильтрации
        self.proxy_model = SortFilterProxyModel()
        self.table_view.setModel(self.proxy_model)
        
        # Статус бар
        self.status_label = QLabel("Готово")
        
        layout.addWidget(toolbar)
        layout.addWidget(self.table_view)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        
        # Подключаем сигналы
        self.filter_btn.triggered.connect(self.open_filter_dialog)
        self.clear_filter_btn.triggered.connect(self.clear_filter)
        
    def set_data(self, df, data_lib='pandas'):
        """Устанавливает данные в модель"""
        if data_lib == 'pandas':
            model = PandasTableModel(df)
        elif data_lib == 'polars':
            model = PolarsTableModel(df)
        else:
            raise ValueError(f"Неподдерживаемая библиотека: {data_lib}")
            
        self.proxy_model.setSourceModel(model)
        self.status_label.setText(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
        
    def open_filter_dialog(self):
        """Открывает диалог фильтрации"""
        if not self.proxy_model.sourceModel():
            return
            
        dialog = FilterDialog(self.proxy_model, self)
        if dialog.exec():
            filters = dialog.get_filters()
            self.apply_filters(filters)
            
    def apply_filters(self, filters):
        """Применяет фильтры к данным"""
        self.proxy_model.set_filters(filters)
        visible_rows = self.proxy_model.rowCount()
        total_rows = self.proxy_model.sourceModel().rowCount()
        self.status_label.setText(
            f"Показано {visible_rows} из {total_rows} строк"
        )
        
    def clear_filter(self):
        """Очищает фильтры"""
        self.proxy_model.set_filters({})
        total_rows = self.proxy_model.sourceModel().rowCount()
        self.status_label.setText(f"Показано {total_rows} строк")


class FilterDialog(QDialog):
    """Диалог настройки фильтров"""
    def __init__(self, proxy_model, parent=None):
        super().__init__(parent)
        self.proxy_model = proxy_model
        self.filters = {}
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Фильтры данных")
        self.setModal(True)
        layout = QVBoxLayout()
        
        source_model = self.proxy_model.sourceModel()
        if not source_model:
            layout.addWidget(QLabel("Нет данных для фильтрации"))
            self.setLayout(layout)
            return
            
        # Получаем информацию о колонках
        self.column_widgets = {}
        
        for col in range(source_model.columnCount()):
            column_name = source_model.headerData(col, Qt.Orientation.Horizontal)
            dtype = source_model.get_column_dtype(col)
            
            group = QGroupBox(column_name)
            group_layout = QVBoxLayout()
            
            # В зависимости от типа данных создаем разные виджеты
            if dtype in ['int64', 'float64', 'int32', 'float32']:
                min_spin = QDoubleSpinBox()
                min_spin.setRange(-1e9, 1e9)
                min_spin.setValue(-1e9)
                max_spin = QDoubleSpinBox()
                max_spin.setRange(-1e9, 1e9)
                max_spin.setValue(1e9)
                
                group_layout.addWidget(QLabel("Мин:"))
                group_layout.addWidget(min_spin)
                group_layout.addWidget(QLabel("Макс:"))
                group_layout.addWidget(max_spin)
                
                self.column_widgets[column_name] = {
                    'type': 'range',
                    'min': min_spin,
                    'max': max_spin,
                    'dtype': dtype
                }
                
            elif dtype == 'object':
                line_edit = QLineEdit()
                line_edit.setPlaceholderText("Текст для поиска...")
                group_layout.addWidget(line_edit)
                self.column_widgets[column_name] = {
                    'type': 'text',
                    'widget': line_edit
                }
                
            elif dtype == 'datetime64[ns]':
                date_from = QDateEdit()
                date_from.setCalendarPopup(True)
                date_from.setDate(QDate.currentDate().addYears(-1))
                date_to = QDateEdit()
                date_to.setCalendarPopup(True)
                date_to.setDate(QDate.currentDate())
                
                group_layout.addWidget(QLabel("С:"))
                group_layout.addWidget(date_from)
                group_layout.addWidget(QLabel("По:"))
                group_layout.addWidget(date_to)
                
                self.column_widgets[column_name] = {
                    'type': 'date',
                    'from': date_from,
                    'to': date_to
                }
                
            elif dtype == 'bool':
                combo = QComboBox()
                combo.addItem("Любое", None)
                combo.addItem("True", True)
                combo.addItem("False", False)
                group_layout.addWidget(combo)
                self.column_widgets[column_name] = {
                    'type': 'bool',
                    'widget': combo
                }
                
            group.setLayout(group_layout)
            layout.addWidget(group)
            
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.resize(400, 600)
        
    def get_filters(self):
        """Возвращает словарь фильтров"""
        filters = {}
        for col_name, widgets in self.column_widgets.items():
            filter_data = {}
            
            if widgets['type'] == 'range':
                min_val = widgets['min'].value()
                max_val = widgets['max'].value()
                if min_val > -1e9 or max_val < 1e9:
                    filter_data = {
                        'type': 'range',
                        'min': float(min_val),
                        'max': float(max_val)
                    }
                    
            elif widgets['type'] == 'text':
                text = widgets['widget'].text().strip()
                if text:
                    filter_data = {
                        'type': 'text',
                        'value': text
                    }
                    
            elif widgets['type'] == 'date':
                date_from = widgets['from'].date().toPyDate()
                date_to = widgets['to'].date().toPyDate()
                filter_data = {
                    'type': 'date',
                    'from': date_from,
                    'to': date_to
                }
                
            elif widgets['type'] == 'bool':
                value = widgets['widget'].currentData()
                if value is not None:
                    filter_data = {
                        'type': 'bool',
                        'value': value
                    }
                    
            if filter_data:
                filters[col_name] = filter_data
                
        return filters


class DashboardTab(QWidget):
    """Вкладка с дашбордом"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Панель управления графиками
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Линейный график",
            "Столбчатая диаграмма",
            "Круговая диаграмма",
            "Точечная диаграмма",
            "Гистограмма"
        ])
        
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        
        self.refresh_btn = QPushButton("Обновить график")
        self.refresh_btn.clicked.connect(self.update_chart)
        
        control_layout.addWidget(QLabel("Тип:"))
        control_layout.addWidget(self.chart_type_combo)
        control_layout.addWidget(QLabel("Ось X:"))
        control_layout.addWidget(self.x_axis_combo)
        control_layout.addWidget(QLabel("Ось Y:"))
        control_layout.addWidget(self.y_axis_combo)
        control_layout.addWidget(self.refresh_btn)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        
        # Виджеты визуализации
        tab_widget = QTabWidget()
        
        # Matplotlib
        self.matplotlib_widget = MatplotlibWidget()
        tab_widget.addTab(self.matplotlib_widget, "Matplotlib")
        
        # Plotly
        self.plotly_widget = PlotlyWidget()
        tab_widget.addTab(self.plotly_widget, "Plotly")
        
        # PyQtGraph
        self.pyqtgraph_widget = PyQtGraphWidget()
        tab_widget.addTab(self.pyqtgraph_widget, "PyQtGraph (Real-time)")
        
        layout.addWidget(control_panel)
        layout.addWidget(tab_widget)
        self.setLayout(layout)
        
    def set_data(self, df):
        """Устанавливает данные и обновляет комбобоксы"""
        self.data = df
        columns = list(df.columns)
        
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        
        self.x_axis_combo.addItems(columns)
        self.y_axis_combo.addItems(columns)
        
        # Выбираем подходящие колонки по умолчанию
        numeric_cols = [col for col in columns if self.is_numeric(df[col])]
        if numeric_cols:
            self.y_axis_combo.setCurrentText(numeric_cols[0])
            
    def is_numeric(self, series):
        """Проверяет, является ли серия числовой"""
        if hasattr(series, 'dtype'):
            return pd.api.types.is_numeric_dtype(series.dtype)
        return False
        
    def update_chart(self):
        """Обновляет все графики"""
        if self.data is None:
            return
            
        chart_type = self.chart_type_combo.currentText()
        x_col = self.x_axis_combo.currentText()
        y_col = self.y_axis_combo.currentText()
        
        # Matplotlib
        self.matplotlib_widget.update_chart(
            self.data, chart_type, x_col, y_col
        )
        
        # Plotly
        fig = create_plotly_figure(self.data, chart_type, x_col, y_col)
        self.plotly_widget.set_figure(fig)
        
        # PyQtGraph
        self.pyqtgraph_widget.update_chart(
            self.data, chart_type, x_col, y_col
        )


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.current_data_lib = 'pandas'
        self.data_processor = DataProcessor()
        self.init_ui()
        self.apply_styles()
        
    def init_ui(self):
        self.setWindowTitle("BI Dashboard - Pandas & Polars")
        self.setGeometry(100, 100, 1400, 800)
        
        # Создаем центральный виджет с вкладками
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Вкладка данных
        self.data_view = DataViewWidget()
        self.tab_widget.addTab(self.data_view, "Данные")
        
        # Вкладка дашборда
        self.dashboard = DashboardTab()
        self.tab_widget.addTab(self.dashboard, "Дашборд")
        
        # Вкладка анализа
        self.analysis_widget = QWidget()
        self.setup_analysis_tab()
        self.tab_widget.addTab(self.analysis_widget, "Анализ")
        
        # Создаем меню
        self.create_menu()
        
        # Создаем статус бар
        self.statusBar().showMessage("Готово")
        
        # Прогресс бар в статус баре
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
    def setup_analysis_tab(self):
        """Настраивает вкладку анализа"""
        layout = QVBoxLayout()
        
        # Кнопки анализа
        analysis_buttons = QWidget()
        btn_layout = QHBoxLayout()
        
        self.analyze_btn = QPushButton("Анализ данных")
        self.correlation_btn = QPushButton("Корреляция")
        self.describe_btn = QPushButton("Статистика")
        self.groupby_btn = QPushButton("Группировка")
        
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.correlation_btn.clicked.connect(self.show_correlation)
        self.describe_btn.clicked.connect(self.show_description)
        self.groupby_btn.clicked.connect(self.show_groupby)
        
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addWidget(self.correlation_btn)
        btn_layout.addWidget(self.describe_btn)
        btn_layout.addWidget(self.groupby_btn)
        btn_layout.addStretch()
        
        analysis_buttons.setLayout(btn_layout)
        
        # Текстовая область для результатов
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        
        layout.addWidget(analysis_buttons)
        layout.addWidget(self.results_text)
        self.analysis_widget.setLayout(layout)
        
    def create_menu(self):
        """Создает меню приложения"""
        menubar = self.menuBar()
        
        # Меню Файл
        file_menu = menubar.addMenu("Файл")
        
        load_action = QAction("Загрузить данные", self)
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)
        
        generate_action = QAction("Сгенерировать тестовые данные", self)
        generate_action.triggered.connect(self.generate_test_data)
        file_menu.addAction(generate_action)
        
        file_menu.addSeparator()
        
        export_menu = file_menu.addMenu("Экспорт")
        
        export_csv_action = QAction("Экспорт в CSV", self)
        export_csv_action.triggered.connect(lambda: self.export_data('csv'))
        export_menu.addAction(export_csv_action)
        
        export_excel_action = QAction("Экспорт в Excel", self)
        export_excel_action.triggered.connect(lambda: self.export_data('excel'))
        export_menu.addAction(export_excel_action)
        
        export_json_action = QAction("Экспорт в JSON", self)
        export_json_action.triggered.connect(lambda: self.export_data('json'))
        export_menu.addAction(export_json_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Меню Данные
        data_menu = menubar.addMenu("Данные")
        
        convert_pandas_action = QAction("Конвертировать в Pandas", self)
        convert_pandas_action.triggered.connect(
            lambda: self.convert_data_lib('pandas')
        )
        data_menu.addAction(convert_pandas_action)
        
        convert_polars_action = QAction("Конвертировать в Polars", self)
        convert_polars_action.triggered.connect(
            lambda: self.convert_data_lib('polars')
        )
        data_menu.addAction(convert_polars_action)
        
        # Меню Настройки
        settings_menu = menubar.addMenu("Настройки")
        
        theme_menu = settings_menu.addMenu("Тема")
        
        light_action = QAction("Светлая", self)
        light_action.triggered.connect(lambda: self.set_theme('light'))
        theme_menu.addAction(light_action)
        
        dark_action = QAction("Темная", self)
        dark_action.triggered.connect(lambda: self.set_theme('dark'))
        theme_menu.addAction(dark_action)
        
        blue_action = QAction("Синяя", self)
        blue_action.triggered.connect(lambda: self.set_theme('blue'))
        theme_menu.addAction(blue_action)
        
    def apply_styles(self):
        """Применяет стили к приложению"""
        StyleManager.apply_style(self, 'light')
        
    def set_theme(self, theme_name):
        """Устанавливает тему приложения"""
        StyleManager.apply_style(self, theme_name)
        
    def load_data(self):
        """Загружает данные из файла"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Выберите файл данных",
            "",
            "Все файлы (*.*);;CSV (*.csv);;Excel (*.xlsx *.xls);;JSON (*.json);;Parquet (*.parquet)"
        )
        
        if file_path:
            self.show_progress("Загрузка данных...")
            
            # Запускаем в отдельном потоке
            self.loader_thread = DataLoaderThread(file_path)
            self.loader_thread.data_loaded.connect(self.on_data_loaded)
            self.loader_thread.error_occurred.connect(self.on_load_error)
            self.loader_thread.progress_updated.connect(self.update_progress)
            self.loader_thread.start()
            
    def generate_test_data(self):
        """Генерирует тестовые данные"""
        dialog = TestDataDialog(self)
        if dialog.exec():
            rows, data_lib = dialog.get_parameters()
            self.show_progress("Генерация тестовых данных...")
            
            # Запускаем генерацию в отдельном потоке
            self.loader_thread = DataLoaderThread(None, rows, data_lib)
            self.loader_thread.data_loaded.connect(self.on_data_loaded)
            self.loader_thread.error_occurred.connect(self.on_load_error)
            self.loader_thread.progress_updated.connect(self.update_progress)
            self.loader_thread.start()
            
    def on_data_loaded(self, result):
        """Обрабатывает загруженные данные"""
        df, data_lib = result
        self.current_data = df
        self.current_data_lib = data_lib
        
        # Обновляем представление данных
        self.data_view.set_data(df, data_lib)
        
        # Обновляем дашборд
        if data_lib == 'polars':
            # Конвертируем в pandas для визуализации
            df_pd = DataConverter.polars_to_pandas(df)
            self.dashboard.set_data(df_pd)
        else:
            self.dashboard.set_data(df)
            
        self.hide_progress("Данные успешно загружены")
        
    def on_load_error(self, error_msg):
        """Обрабатывает ошибку загрузки"""
        self.hide_progress("")
        QMessageBox.critical(self, "Ошибка", error_msg)
        
    def convert_data_lib(self, target_lib):
        """Конвертирует данные между библиотеками"""
        if self.current_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для конвертации")
            return
            
        self.show_progress(f"Конвертация в {target_lib}...")
        
        # Используем QTimer для имитации асинхронной операции
        QTimer.singleShot(100, lambda: self.perform_conversion(target_lib))
        
    def perform_conversion(self, target_lib):
        """Выполняет конвертацию данных"""
        try:
            if self.current_data_lib == target_lib:
                self.hide_progress("Данные уже в указанном формате")
                return
                
            if target_lib == 'pandas':
                df = DataConverter.polars_to_pandas(self.current_data)
            else:  # polars
                df = DataConverter.pandas_to_polars(self.current_data)
                
            self.current_data = df
            self.current_data_lib = target_lib
            
            # Обновляем представление
            self.data_view.set_data(df, target_lib)
            self.hide_progress(f"Данные конвертированы в {target_lib}")
            
        except Exception as e:
            self.hide_progress("")
            QMessageBox.critical(self, "Ошибка конвертации", str(e))
            
    def export_data(self, format_type):
        """Экспортирует данные в файл"""
        if self.current_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
            return
            
        # Выбор файла для сохранения
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        
        if format_type == 'csv':
            file_path, _ = file_dialog.getSaveFileName(
                self, "Экспорт в CSV", "", "CSV файлы (*.csv)"
            )
            ext = '.csv'
        elif format_type == 'excel':
            file_path, _ = file_dialog.getSaveFileName(
                self, "Экспорт в Excel", "", "Excel файлы (*.xlsx)"
            )
            ext = '.xlsx'
        else:  # json
            file_path, _ = file_dialog.getSaveFileName(
                self, "Экспорт в JSON", "", "JSON файлы (*.json)"
            )
            ext = '.json'
            
        if file_path:
            if not file_path.endswith(ext):
                file_path += ext
                
            self.show_progress(f"Экспорт в {format_type}...")
            
            # Запускаем экспорт в отдельном потоке
            self.export_worker = ExportWorker(
                self.current_data, 
                self.current_data_lib,
                file_path, 
                format_type
            )
            self.export_worker.finished.connect(self.on_export_finished)
            self.export_worker.error.connect(self.on_export_error)
            self.export_worker.start()
            
    def on_export_finished(self):
        """Обрабатывает завершение экспорта"""
        self.hide_progress("Экспорт завершен успешно")
        QMessageBox.information(self, "Экспорт", "Данные успешно экспортированы")
        
    def on_export_error(self, error_msg):
        """Обрабатывает ошибку экспорта"""
        self.hide_progress("")
        QMessageBox.critical(self, "Ошибка экспорта", error_msg)
        
    def run_analysis(self):
        """Выполняет анализ данных"""
        if self.current_data is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для анализа")
            return
            
        try:
            if self.current_data_lib == 'pandas':
                result = self.data_processor.analyze_pandas(self.current_data)
            else:
                result = self.data_processor.analyze_polars(self.current_data)
                
            self.results_text.setText(result)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", str(e))
            
    def show_correlation(self):
        """Показывает матрицу корреляций"""
        if self.current_data is None:
            return
            
        try:
            if self.current_data_lib == 'pandas':
                corr = self.data_processor.get_correlation_pandas(self.current_data)
            else:
                corr = self.data_processor.get_correlation_polars(self.current_data)
                
            self.results_text.setText(str(corr))
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            
    def show_description(self):
        """Показывает статистическое описание"""
        if self.current_data is None:
            return
            
        try:
            if self.current_data_lib == 'pandas':
                desc = self.data_processor.describe_pandas(self.current_data)
            else:
                desc = self.data_processor.describe_polars(self.current_data)
                
            self.results_text.setText(str(desc))
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            
    def show_groupby(self):
        """Показывает группировку данных"""
        if self.current_data is None:
            return
            
        # Диалог для выбора колонок группировки
        dialog = GroupByDialog(list(self.current_data.columns), self)
        if dialog.exec():
            group_col, agg_col, agg_func = dialog.get_parameters()
            
            try:
                if self.current_data_lib == 'pandas':
                    result = self.data_processor.groupby_pandas(
                        self.current_data, group_col, agg_col, agg_func
                    )
                else:
                    result = self.data_processor.groupby_polars(
                        self.current_data, group_col, agg_col, agg_func
                    )
                    
                self.results_text.setText(str(result))
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка группировки", str(e))
                
    def show_progress(self, message):
        """Показывает прогресс бар"""
        self.statusBar().showMessage(message)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # индикатор без определенного значения
        
    def hide_progress(self, message):
        """Скрывает прогресс бар"""
        self.statusBar().showMessage(message)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        
    def update_progress(self, value):
        """Обновляет значение прогресс бара"""
        if value >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)


class TestDataDialog(QDialog):
    """Диалог для генерации тестовых данных"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Генерация тестовых данных")
        layout = QVBoxLayout()
        
        # Количество строк
        rows_layout = QHBoxLayout()
        rows_layout.addWidget(QLabel("Количество строк:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(100, 1000000)
        self.rows_spin.setValue(10000)
        rows_layout.addWidget(self.rows_spin)
        rows_layout.addStretch()
        
        # Библиотека данных
        lib_layout = QHBoxLayout()
        lib_layout.addWidget(QLabel("Библиотека:"))
        self.lib_combo = QComboBox()
        self.lib_combo.addItems(['pandas', 'polars'])
        lib_layout.addWidget(self.lib_combo)
        lib_layout.addStretch()
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addLayout(rows_layout)
        layout.addLayout(lib_layout)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def get_parameters(self):
        """Возвращает параметры генерации"""
        return self.rows_spin.value(), self.lib_combo.currentText()


class GroupByDialog(QDialog):
    """Диалог для настройки группировки"""
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Группировка данных")
        layout = QVBoxLayout()
        
        # Колонка для группировки
        group_layout = QHBoxLayout()
        group_layout.addWidget(QLabel("Группировать по:"))
        self.group_combo = QComboBox()
        self.group_combo.addItems(self.columns)
        group_layout.addWidget(self.group_combo)
        group_layout.addStretch()
        
        # Колонка для агрегации
        agg_col_layout = QHBoxLayout()
        agg_col_layout.addWidget(QLabel("Агрегировать:"))
        self.agg_col_combo = QComboBox()
        self.agg_col_combo.addItems(self.columns)
        agg_col_layout.addWidget(self.agg_col_combo)
        agg_col_layout.addStretch()
        
        # Функция агрегации
        agg_func_layout = QHBoxLayout()
        agg_func_layout.addWidget(QLabel("Функция:"))
        self.agg_func_combo = QComboBox()
        self.agg_func_combo.addItems(['sum', 'mean', 'count', 'min', 'max', 'std'])
        agg_func_layout.addWidget(self.agg_func_combo)
        agg_func_layout.addStretch()
        
        # Кнопки
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addLayout(group_layout)
        layout.addLayout(agg_col_layout)
        layout.addLayout(agg_func_layout)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def get_parameters(self):
        """Возвращает параметры группировки"""
        return (
            self.group_combo.currentText(),
            self.agg_col_combo.currentText(),
            self.agg_func_combo.currentText()
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BI Dashboard")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()