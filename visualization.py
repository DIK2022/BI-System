# visualization.py
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWebEngineWidgets import QWebEngineView
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
import pyqtgraph as pg
import pandas as pd
import numpy as np
import tempfile
import os


class MatplotlibWidget(QWidget):
    """Виджет для Matplotlib графиков"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def update_chart(self, data, chart_type, x_col, y_col):
        """Обновляет график"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        try:
            if chart_type == "Линейный график":
                ax.plot(data[x_col], data[y_col], marker='o', linestyle='-')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Линейный график: {y_col} по {x_col}")
                ax.grid(True, alpha=0.3)
                
            elif chart_type == "Столбчатая диаграмма":
                # Для столбчатой диаграммы группируем по x_col
                if data[x_col].dtype in ['object', 'category']:
                    grouped = data.groupby(x_col)[y_col].mean()
                    x = grouped.index
                    y = grouped.values
                    ax.bar(x, y)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(f"Среднее {y_col}")
                    ax.set_title(f"Столбчатая диаграмма: {y_col} по {x_col}")
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
            elif chart_type == "Круговая диаграмма":
                if data[x_col].dtype in ['object', 'category']:
                    grouped = data.groupby(x_col)[y_col].sum()
                    ax.pie(grouped.values, labels=grouped.index, autopct='%1.1f%%')
                    ax.set_title(f"Круговая диаграмма: {y_col} по {x_col}")
                    
            elif chart_type == "Точечная диаграмма":
                ax.scatter(data[x_col], data[y_col], alpha=0.5)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Точечная диаграмма: {y_col} vs {x_col}")
                ax.grid(True, alpha=0.3)
                
            elif chart_type == "Гистограмма":
                ax.hist(data[y_col], bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel(y_col)
                ax.set_ylabel('Частота')
                ax.set_title(f"Гистограмма распределения {y_col}")
                ax.grid(True, alpha=0.3)
                
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Ошибка: {str(e)}", 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            self.canvas.draw()


class PlotlyWidget(QWebEngineView):
    """Виджет для Plotly графиков"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.temp_files = []
        
    def set_figure(self, fig):
        """Устанавливает Plotly график"""
        # Удаляем старые временные файлы
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
                
        self.temp_files.clear()
        
        # Создаем временный HTML файл
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.html', delete=False, encoding='utf-8'
        )
        fig.write_html(temp_file.name, include_plotlyjs='cdn')
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        
        # Загружаем HTML в WebEngineView
        self.load(QUrl.fromLocalFile(temp_file.name))
        
    def cleanup(self):
        """Очищает временные файлы"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


class PyQtGraphWidget(QWidget):
    """Виджет для PyQtGraph (режим реального времени)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.data = None
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # График
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Элементы управления реальным временем
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        self.realtime_check = QCheckBox("Режим реального времени")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(100, 5000)
        self.interval_spin.setValue(1000)
        self.interval_spin.setSuffix(" мс")
        
        control_layout.addWidget(self.realtime_check)
        control_layout.addWidget(QLabel("Интервал:"))
        control_layout.addWidget(self.interval_spin)
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        
        layout.addWidget(control_panel)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        
        # Таймер для обновления в реальном времени
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_realtime_data)
        self.realtime_check.stateChanged.connect(self.toggle_realtime)
        
    def update_chart(self, data, chart_type, x_col, y_col):
        """Обновляет график"""
        self.data = data
        self.chart_type = chart_type
        self.x_col = x_col
        self.y_col = y_col
        
        self.plot_widget.clear()
        
        try:
            if chart_type in ["Линейный график", "Точечная диаграмма"]:
                x = data[x_col].values
                y = data[y_col].values
                
                if chart_type == "Линейный график":
                    self.plot_widget.plot(x, y, pen=pg.mkPen('b', width=2), 
                                         symbol='o', symbolSize=5)
                else:  # Точечная диаграмма
                    self.plot_widget.plot(x, y, pen=None, 
                                         symbol='o', symbolSize=5)
                                         
                self.plot_widget.setLabel('bottom', x_col)
                self.plot_widget.setLabel('left', y_col)
                self.plot_widget.setTitle(f"{chart_type}: {y_col} по {x_col}")
                
            elif chart_type == "Гистограмма":
                y, x = np.histogram(data[y_col].dropna().values, bins=30)
                self.plot_widget.plot(x, y, stepMode=True, fillLevel=0, 
                                     brush=(0, 0, 255, 150))
                self.plot_widget.setLabel('bottom', y_col)
                self.plot_widget.setLabel('left', 'Частота')
                self.plot_widget.setTitle(f"Гистограмма: {y_col}")
                
        except Exception as e:
            print(f"Ошибка PyQtGraph: {e}")
            
    def toggle_realtime(self, state):
        """Включает/выключает режим реального времени"""
        if state == Qt.CheckState.Checked.value:
            self.timer.start(self.interval_spin.value())
        else:
            self.timer.stop()
            
    def update_realtime_data(self):
        """Обновляет данные в реальном времени"""
        if self.data is not None and hasattr(self, 'chart_type'):
            # Добавляем новые случайные точки
            new_data = self.data.copy()
            
            if hasattr(new_data, 'to_pandas'):
                new_data = new_data.to_pandas()
                
            # Добавляем случайные данные
            last_x = new_data[self.x_col].iloc[-1]
            if isinstance(last_x, (pd.Timestamp, np.datetime64)):
                new_x = last_x + pd.Timedelta(hours=1)
            else:
                new_x = last_x + 1
                
            new_y = np.random.normal(
                new_data[self.y_col].mean(),
                new_data[self.y_col].std()
            )
            
            # Добавляем новую точку
            new_row = {self.x_col: new_x, self.y_col: new_y}
            new_data = pd.concat([new_data, pd.DataFrame([new_row])], ignore_index=True)
            
            # Обновляем график
            self.update_chart(new_data, self.chart_type, self.x_col, self.y_col)
            self.data = new_data


def create_plotly_figure(data, chart_type, x_col, y_col):
    """Создает Plotly график"""
    if hasattr(data, 'to_pandas'):
        data = data.to_pandas()
        
    try:
        if chart_type == "Линейный график":
            fig = px.line(data, x=x_col, y=y_col, 
                         title=f"Линейный график: {y_col} по {x_col}")
            
        elif chart_type == "Столбчатая диаграмма":
            if data[x_col].dtype in ['object', 'category']:
                grouped = data.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(grouped, x=x_col, y=y_col,
                           title=f"Столбчатая диаграмма: {y_col} по {x_col}")
            else:
                fig = px.histogram(data, x=x_col, y=y_col, histfunc='avg',
                                 title=f"Столбчатая диаграмма: {y_col} по {x_col}")
                
        elif chart_type == "Круговая диаграмма":
            if data[x_col].dtype in ['object', 'category']:
                grouped = data.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(grouped, values=y_col, names=x_col,
                           title=f"Круговая диаграмма: {y_col} по {x_col}")
            else:
                fig = px.pie(data, values=y_col, names=x_col,
                           title=f"Круговая диаграмма: {y_col} по {x_col}")
                           
        elif chart_type == "Точечная диаграмма":
            fig = px.scatter(data, x=x_col, y=y_col,
                           title=f"Точечная диаграмма: {y_col} vs {x_col}")
            
        elif chart_type == "Гистограмма":
            fig = px.histogram(data, x=y_col,
                             title=f"Гистограмма распределения {y_col}")
            
        else:
            fig = px.scatter(data, x=x_col, y=y_col)
            
        fig.update_layout(
            template="plotly_white",
            hovermode="closest"
        )
        
        return fig
        
    except Exception as e:
        # Создаем пустой график с сообщением об ошибке
        fig = go.Figure()
        fig.add_annotation(
            text=f"Ошибка: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig