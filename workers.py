# workers.py
from PyQt6.QtCore import *
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import json


class DataLoaderThread(QThread):
    """Поток для загрузки данных"""
    data_loaded = pyqtSignal(tuple)  # (dataframe, data_lib)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, file_path=None, rows=10000, data_lib='pandas'):
        super().__init__()
        self.file_path = file_path
        self.rows = rows
        self.data_lib = data_lib
        
    def run(self):
        try:
            if self.file_path:
                # Загрузка из файла
                self.progress_updated.emit(10)
                
                if self.file_path.endswith('.csv'):
                    if self.data_lib == 'pandas':
                        df = pd.read_csv(self.file_path)
                    else:
                        df = pl.read_csv(self.file_path)
                        
                elif self.file_path.endswith('.parquet'):
                    if self.data_lib == 'pandas':
                        df = pd.read_parquet(self.file_path)
                    else:
                        df = pl.read_parquet(self.file_path)
                        
                elif self.file_path.endswith(('.xlsx', '.xls')):
                    # Excel поддерживается только pandas
                    df = pd.read_excel(self.file_path)
                    if self.data_lib == 'polars':
                        df = pl.from_pandas(df)
                        
                elif self.file_path.endswith('.json'):
                    if self.data_lib == 'pandas':
                        df = pd.read_json(self.file_path)
                    else:
                        df = pl.read_json(self.file_path)
                        
                else:
                    # Пробуем загрузить как CSV
                    if self.data_lib == 'pandas':
                        df = pd.read_csv(self.file_path)
                    else:
                        df = pl.read_csv(self.file_path)
                        
            else:
                # Генерация тестовых данных
                df = self.generate_test_data()
                
            self.progress_updated.emit(100)
            self.data_loaded.emit((df, self.data_lib))
            
        except Exception as e:
            self.error_occurred.emit(f"Ошибка загрузки: {str(e)}")
            
    def generate_test_data(self):
        """Генерирует тестовые данные"""
        np.random.seed(42)
        
        # Генерируем даты
        dates = [datetime.now() - timedelta(days=i) for i in range(self.rows)]
        
        # Генерируем разные типы данных
        data = {
            'date': dates,
            'category': np.random.choice(['A', 'B', 'C', 'D'], self.rows),
            'value_int': np.random.randint(1, 1000, self.rows),
            'value_float': np.random.uniform(0, 1000, self.rows),
            'sales': np.random.exponential(100, self.rows),
            'profit': np.random.normal(500, 200, self.rows),
            'region': np.random.choice(['North', 'South', 'East', 'West'], self.rows),
            'active': np.random.choice([True, False], self.rows),
            'score': np.random.randint(1, 101, self.rows)
        }
        
        if self.data_lib == 'pandas':
            df = pd.DataFrame(data)
        else:
            df = pl.DataFrame(data)
            
        return df


class ExportWorker(QThread):
    """Поток для экспорта данных"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, data, data_lib, file_path, format_type):
        super().__init__()
        self.data = data
        self.data_lib = data_lib
        self.file_path = file_path
        self.format_type = format_type
        
    def run(self):
        try:
            if self.format_type == 'csv':
                if self.data_lib == 'pandas':
                    self.data.to_csv(self.file_path, index=False)
                else:
                    self.data.write_csv(self.file_path)
                    
            elif self.format_type == 'excel':
                # Для Excel используем pandas
                if self.data_lib == 'pandas':
                    self.data.to_excel(self.file_path, index=False)
                else:
                    df_pd = self.data.to_pandas()
                    df_pd.to_excel(self.file_path, index=False)
                    
            elif self.format_type == 'json':
                if self.data_lib == 'pandas':
                    self.data.to_json(self.file_path, orient='records', indent=2)
                else:
                    self.data.write_json(self.file_path)
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Ошибка экспорта: {str(e)}")