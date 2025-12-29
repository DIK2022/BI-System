# data_models.py
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime


class PandasTableModel(QAbstractTableModel):
    """Модель таблицы для Pandas DataFrame"""
    
    def __init__(self, data):
        super().__init__()
        self._data = data.copy() if data is not None else pd.DataFrame()
        self._original_data = data.copy() if data is not None else pd.DataFrame()
        
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self._data.columns)
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            
            # Форматирование различных типов данных
            if pd.isna(value):
                return ""
            elif isinstance(value, (np.integer, np.floating)):
                # Для больших чисел добавляем разделители тысяч
                if isinstance(value, np.integer):
                    return f"{value:,}"
                else:
                    return f"{value:,.2f}"
            elif isinstance(value, datetime):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(value, pd.Timestamp):
                return value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                return str(value)
                
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (int, float, np.integer, np.floating)):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Цветовая схема для числовых значений
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Градиент от синего к красному
                normalized = abs(value) / (self._data.iloc[:, index.column()].abs().max() + 1e-9)
                red = min(255, int(255 * normalized))
                blue = 255 - red
                return QColor(red, 200, blue)
                
        elif role == Qt.ItemDataRole.ToolTipRole:
            value = self._data.iloc[index.row(), index.column()]
            return f"Тип: {type(value).__name__}\nЗначение: {value}"
            
        return None
        
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
            
        if orientation == Qt.Orientation.Horizontal:
            return str(self._data.columns[section])
        else:
            return str(section + 1)
            
    def get_column_dtype(self, column):
        """Возвращает тип данных колонки"""
        if column < self.columnCount():
            dtype = str(self._data.dtypes[column])
            return dtype
        return 'object'
        
    def sort(self, column, order):
        """Сортировка данных"""
        self.layoutAboutToBeChanged.emit()
        
        col_name = self._data.columns[column]
        ascending = (order == Qt.SortOrder.AscendingOrder)
        
        try:
            self._data = self._data.sort_values(by=col_name, ascending=ascending)
        except Exception as e:
            print(f"Ошибка сортировки: {e}")
            
        self.layoutChanged.emit()
        
    def get_dataframe(self):
        """Возвращает DataFrame"""
        return self._data.copy()


class PolarsTableModel(QAbstractTableModel):
    """Модель таблицы для Polars DataFrame"""
    
    def __init__(self, data):
        super().__init__()
        self._data = data.clone() if data is not None else pl.DataFrame()
        self._original_data = data.clone() if data is not None else pl.DataFrame()
        
    def rowCount(self, parent=None):
        return self._data.height
    
    def columnCount(self, parent=None):
        return self._data.width
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.ItemDataRole.DisplayRole:
            row = index.row()
            col = index.column()
            
            if row < self.rowCount() and col < self.columnCount():
                value = self._data[row, col]
                
                # Обработка null значений
                if value is None:
                    return ""
                    
                # Форматирование
                col_name = self._data.columns[col]
                dtype = self._data[col_name].dtype
                
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                    return f"{value:,}"
                elif dtype in [pl.Float32, pl.Float64]:
                    return f"{value:,.2f}"
                elif dtype == pl.Date:
                    return value.strftime("%Y-%m-%d")
                elif dtype == pl.Datetime:
                    return value.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    return str(value)
                    
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            col_name = self._data.columns[index.column()]
            dtype = self._data[col_name].dtype
            
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                
        elif role == Qt.ItemDataRole.BackgroundRole:
            # Цветовая схема
            row = index.row()
            col = index.column()
            
            if row < self.rowCount() and col < self.columnCount():
                value = self._data[row, col]
                col_name = self._data.columns[col]
                dtype = self._data[col_name].dtype
                
                if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                           pl.Float32, pl.Float64] and value is not None:
                    
                    # Получаем максимальное значение в колонке
                    max_val = self._data[col_name].max()
                    min_val = self._data[col_name].min()
                    
                    if max_val != min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                        red = min(255, int(255 * normalized))
                        blue = 255 - red
                        return QColor(red, 200, blue)
                        
        elif role == Qt.ItemDataRole.ToolTipRole:
            row = index.row()
            col = index.column()
            
            if row < self.rowCount() and col < self.columnCount():
                value = self._data[row, col]
                col_name = self._data.columns[col]
                dtype = self._data[col_name].dtype
                
                return f"Колонка: {col_name}\nТип: {dtype}\nЗначение: {value}"
                
        return None
        
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
            
        if orientation == Qt.Orientation.Horizontal:
            return str(self._data.columns[section])
        else:
            return str(section + 1)
            
    def get_column_dtype(self, column):
        """Возвращает тип данных колонки"""
        if column < self.columnCount():
            col_name = self._data.columns[column]
            dtype = str(self._data[col_name].dtype)
            # Приводим к общим типам для совместимости
            if 'Int' in dtype or 'UInt' in dtype:
                return 'int64'
            elif 'Float' in dtype:
                return 'float64'
            elif 'Date' in dtype:
                return 'datetime64[ns]'
            else:
                return 'object'
        return 'object'
        
    def sort(self, column, order):
        """Сортировка данных"""
        self.layoutAboutToBeChanged.emit()
        
        col_name = self._data.columns[column]
        descending = (order == Qt.SortOrder.DescendingOrder)
        
        try:
            self._data = self._data.sort(col_name, descending=descending)
        except Exception as e:
            print(f"Ошибка сортировки: {e}")
            
        self.layoutChanged.emit()
        
    def get_dataframe(self):
        """Возвращает DataFrame"""
        return self._data.clone()


class SortFilterProxyModel(QSortFilterProxyModel):
    """Прокси-модель для фильтрации данных"""
    
    def __init__(self):
        super().__init__()
        self._filters = {}
        
    def set_filters(self, filters):
        """Устанавливает фильтры"""
        self._filters = filters
        self.invalidateFilter()
        
    def filterAcceptsRow(self, source_row, source_parent):
        """Проверяет, проходит ли строка фильтрацию"""
        source_model = self.sourceModel()
        if not source_model:
            return True
            
        # Применяем все фильтры
        for col_name, filter_data in self._filters.items():
            # Находим индекс колонки
            col_index = -1
            for i in range(source_model.columnCount()):
                if source_model.headerData(i, Qt.Orientation.Horizontal) == col_name:
                    col_index = i
                    break
                    
            if col_index == -1:
                continue
                
            # Получаем значение ячейки
            index = source_model.index(source_row, col_index, source_parent)
            value = source_model.data(index, Qt.ItemDataRole.DisplayRole)
            
            # Применяем фильтр в зависимости от типа
            if not self.apply_filter(value, filter_data):
                return False
                
        return True
        
    def apply_filter(self, value, filter_data):
        """Применяет конкретный фильтр к значению"""
        filter_type = filter_data.get('type')
        
        try:
            if filter_type == 'range':
                min_val = filter_data.get('min', -float('inf'))
                max_val = filter_data.get('max', float('inf'))
                
                # Пытаемся преобразовать значение в число
                try:
                    num_val = float(str(value).replace(',', ''))
                    return min_val <= num_val <= max_val
                except (ValueError, TypeError):
                    return False
                    
            elif filter_type == 'text':
                search_text = filter_data.get('value', '').lower()
                if not search_text:
                    return True
                return search_text in str(value).lower()
                
            elif filter_type == 'date':
                date_from = filter_data.get('from')
                date_to = filter_data.get('to')
                
                try:
                    # Пытаемся разобрать дату
                    if isinstance(value, str):
                        # Пробуем разные форматы дат
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y"]:
                            try:
                                date_val = datetime.strptime(value, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            return False
                    else:
                        date_val = value
                        
                    return date_from <= date_val.date() <= date_to
                except (ValueError, TypeError, AttributeError):
                    return False
                    
            elif filter_type == 'bool':
                filter_value = filter_data.get('value')
                value_str = str(value).lower()
                
                if filter_value is True:
                    return value_str in ['true', '1', 'да', 'yes']
                elif filter_value is False:
                    return value_str in ['false', '0', 'нет', 'no']
                    
        except Exception:
            return True  # В случае ошибки пропускаем строку
            
        return True